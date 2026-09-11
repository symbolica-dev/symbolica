//! Diagnostic-only capture; enable `polynomial_benchmark_capture` and set
//! SYMBOLICA_POLY_CAPTURE_DIR. Timings are inclusive and are not benchmark results.

use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use super::{Exponent, polynomial::MultivariatePolynomial};
use crate::domains::Ring;

static CONFIG: OnceLock<Option<(PathBuf, Duration)>> = OnceLock::new();
static NEXT: AtomicUsize = AtomicUsize::new(0);

pub(super) struct Capture<'a, R: Ring, E: Exponent> {
    operation: &'static str,
    left: &'a MultivariatePolynomial<R, E>,
    right: &'a MultivariatePolynomial<R, E>,
    start: Instant,
    directory: &'static PathBuf,
    threshold: Duration,
}

impl<'a, R: Ring, E: Exponent> Capture<'a, R, E> {
    pub(super) fn start(
        operation: &'static str,
        left: &'a MultivariatePolynomial<R, E>,
        right: &'a MultivariatePolynomial<R, E>,
    ) -> Option<Self> {
        let (directory, threshold) = CONFIG
            .get_or_init(|| {
                let directory = PathBuf::from(std::env::var_os("SYMBOLICA_POLY_CAPTURE_DIR")?);
                fs::create_dir_all(&directory).expect("create operand capture directory");
                let millis = std::env::var("SYMBOLICA_POLY_CAPTURE_MS")
                    .map(|s| s.parse::<u64>().expect("capture threshold in milliseconds"))
                    .unwrap_or(100);
                Some((directory, Duration::from_millis(millis)))
            })
            .as_ref()?;
        // The standalone replay harness uses Z. Do not accidentally export modular
        // or polynomial-valued coefficients as integer benchmark inputs.
        if std::any::type_name::<R>()
            != std::any::type_name::<crate::domains::integer::IntegerRing>()
            || left.variables() != right.variables()
        {
            return None;
        }
        Some(Self {
            operation,
            left,
            right,
            start: Instant::now(),
            directory,
            threshold: *threshold,
        })
    }
}

impl<R: Ring, E: Exponent> Drop for Capture<'_, R, E> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if elapsed < self.threshold || std::thread::panicking() {
            return;
        }
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let stem = format!("{}-{id:04}", self.operation);
        for (side, poly) in [("left", self.left), ("right", self.right)] {
            let file = File::create(self.directory.join(format!("{stem}.{side}.txt.br"))).unwrap();
            let mut out = brotli::CompressorWriter::new(file, 65536, 5, 22);
            // Positional variable names preserve variable order independently of namespaces.
            for (i, term) in poly.into_iter().enumerate() {
                let coefficient = poly.ring().printer(term.coefficient).to_string();
                if i != 0 && !coefficient.starts_with('-') {
                    write!(out, "+").unwrap();
                }
                write!(out, "{coefficient}").unwrap();
                for (v, e) in term.exponents.iter().enumerate() {
                    if *e != E::zero() {
                        write!(out, "*x{v}^{}", e.to_i32()).unwrap();
                    }
                }
            }
            if poly.is_zero() {
                write!(out, "0").unwrap();
            }
            writeln!(out).unwrap();
        }
        fs::write(
            self.directory.join(format!("{stem}.meta")),
            format!(
                "operation={}\nvariables={}\nleft_terms={}\nright_terms={}\nseconds={:.6}\n",
                self.operation,
                self.left.nvars(),
                self.left.nterms(),
                self.right.nterms(),
                elapsed.as_secs_f64()
            ),
        )
        .unwrap();
        eprintln!(
            "captured {stem}: {} x {} terms, {:.6}s",
            self.left.nterms(),
            self.right.nterms(),
            elapsed.as_secs_f64()
        );
    }
}
