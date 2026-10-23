use std::{
    env,
    hint::black_box,
    sync::Once,
    time::{Duration, Instant},
};

/// Environment variable that overrides the number of measured samples per backend.
pub const SAMPLES_ENV: &str = "SYMBOLICA_FLINT_BENCH_SAMPLES";

/// Environment variable that selects cases whose names contain the supplied text.
pub const FILTER_ENV: &str = "SYMBOLICA_FLINT_BENCH_FILTER";

/// Environment variable that selects CSV output when set to a true boolean value.
pub const CSV_ENV: &str = "SYMBOLICA_FLINT_BENCH_CSV";

static CSV_HEADER: Once = Once::new();

/// Runtime controls shared by the paired Symbolica and FLINT benchmarks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedConfig {
    samples: usize,
    filter: Option<String>,
    csv: bool,
}

impl PairedConfig {
    /// Creates controls for `samples` measurements of each backend.
    pub fn new(samples: usize) -> Self {
        assert!(
            samples > 0,
            "paired benchmark sample count must be positive"
        );
        Self {
            samples,
            filter: None,
            csv: false,
        }
    }

    /// Reads benchmark controls from the `SYMBOLICA_FLINT_BENCH_*` environment variables.
    ///
    /// `SYMBOLICA_FLINT_BENCH_SAMPLES` overrides `default_samples`,
    /// `SYMBOLICA_FLINT_BENCH_FILTER` selects case names by substring, and
    /// `SYMBOLICA_FLINT_BENCH_CSV` selects CSV instead of human-readable output.
    pub fn from_env(default_samples: usize) -> Self {
        let mut config = Self::new(default_samples);

        if let Some(value) = env::var_os(SAMPLES_ENV) {
            let value = value
                .into_string()
                .unwrap_or_else(|_| panic!("{SAMPLES_ENV} must contain valid UTF-8"));
            config.samples = value
                .parse::<usize>()
                .ok()
                .filter(|samples| *samples > 0)
                .unwrap_or_else(|| panic!("{SAMPLES_ENV} must be a positive integer"));
        }

        config.filter = env::var(FILTER_ENV)
            .ok()
            .and_then(|filter| (!filter.is_empty()).then_some(filter));

        if let Some(value) = env::var_os(CSV_ENV) {
            let value = value
                .into_string()
                .unwrap_or_else(|_| panic!("{CSV_ENV} must contain valid UTF-8"));
            config.csv = parse_bool(CSV_ENV, &value);
        }

        config
    }

    /// Returns the number of measured samples taken for each backend.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Returns whether `name` is selected by the configured substring filter.
    pub fn matches(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
    }
}

/// Minimum and median elapsed time for one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimingSummary {
    pub min: Duration,
    pub median: Duration,
}

impl TimingSummary {
    /// Returns the minimum time in milliseconds.
    pub fn min_millis(self) -> f64 {
        self.min.as_secs_f64() * 1_000.0
    }

    /// Returns the median time in milliseconds.
    pub fn median_millis(self) -> f64 {
        self.median.as_secs_f64() * 1_000.0
    }
}

/// Timings collected for a single case from both implementations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedResult {
    pub name: String,
    pub samples: usize,
    pub symbolica: TimingSummary,
    pub flint: TimingSummary,
}

impl PairedResult {
    /// Returns the ratio of Symbolica's median time to FLINT's median time.
    pub fn symbolica_over_flint(&self) -> f64 {
        self.symbolica.median.as_secs_f64() / self.flint.median.as_secs_f64()
    }
}

/// Measures matching Symbolica and FLINT operations and prints their timing row.
///
/// Each operation is called once before measurement. Measured samples alternate
/// between Symbolica-first and FLINT-first order. The returned output of an
/// operation is retained until its timer has stopped and is then dropped.
pub fn run_paired<Symbolica, Flint, SymbolicaOutput, FlintOutput>(
    config: &PairedConfig,
    name: &str,
    mut symbolica: Symbolica,
    mut flint: Flint,
) -> Option<PairedResult>
where
    Symbolica: FnMut() -> SymbolicaOutput,
    Flint: FnMut() -> FlintOutput,
{
    if !config.matches(name) {
        return None;
    }

    drop(black_box(symbolica()));
    drop(black_box(flint()));

    let mut symbolica_samples = Vec::with_capacity(config.samples);
    let mut flint_samples = Vec::with_capacity(config.samples);
    for sample in 0..config.samples {
        if sample % 2 == 0 {
            symbolica_samples.push(measure_once(&mut symbolica));
            flint_samples.push(measure_once(&mut flint));
        } else {
            flint_samples.push(measure_once(&mut flint));
            symbolica_samples.push(measure_once(&mut symbolica));
        }
    }

    let result = PairedResult {
        name: name.to_owned(),
        samples: config.samples,
        symbolica: summarize(symbolica_samples),
        flint: summarize(flint_samples),
    };
    print_result(&result, config.csv);
    Some(result)
}

fn measure_once<Operation, Output>(operation: &mut Operation) -> Duration
where
    Operation: FnMut() -> Output,
{
    let start = Instant::now();
    let output = operation();
    let elapsed = start.elapsed();
    drop(black_box(output));
    elapsed
}

fn summarize(mut samples: Vec<Duration>) -> TimingSummary {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    let median = if samples.len() % 2 == 0 {
        average_duration(samples[middle - 1], samples[middle])
    } else {
        samples[middle]
    };
    TimingSummary {
        min: samples[0],
        median,
    }
}

fn average_duration(left: Duration, right: Duration) -> Duration {
    let nanos = (left.as_nanos() + right.as_nanos()) / 2;
    Duration::new(
        (nanos / 1_000_000_000) as u64,
        (nanos % 1_000_000_000) as u32,
    )
}

fn print_result(result: &PairedResult, csv: bool) {
    if csv {
        CSV_HEADER.call_once(|| {
            println!(
                "case,samples,symbolica_min_ms,symbolica_median_ms,flint_min_ms,flint_median_ms,symbolica_over_flint"
            );
        });
        println!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            csv_field(&result.name),
            result.samples,
            result.symbolica.min_millis(),
            result.symbolica.median_millis(),
            result.flint.min_millis(),
            result.flint.median_millis(),
            result.symbolica_over_flint(),
        );
    } else {
        println!(
            "{:<48} samples {:>3}  Symbolica min/median {:>10.3}/{:>10.3} ms  FLINT min/median {:>10.3}/{:>10.3} ms  S/F {:>7.3}x",
            result.name,
            result.samples,
            result.symbolica.min_millis(),
            result.symbolica.median_millis(),
            result.flint.min_millis(),
            result.flint.median_millis(),
            result.symbolica_over_flint(),
        );
    }
}

fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn parse_bool(variable: &str, value: &str) -> bool {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => panic!("{variable} must be one of 1, 0, true, false, yes, no, on, or off"),
    }
}

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    struct DropLog<'a> {
        event: &'static str,
        log: &'a RefCell<Vec<&'static str>>,
    }

    impl Drop for DropLog<'_> {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.event);
        }
    }

    #[test]
    fn even_sample_median_averages_middle_values() {
        let summary = summarize(vec![
            Duration::from_millis(8),
            Duration::from_millis(2),
            Duration::from_millis(6),
            Duration::from_millis(4),
        ]);
        assert_eq!(summary.min, Duration::from_millis(2));
        assert_eq!(summary.median, Duration::from_millis(5));
    }

    #[test]
    fn filter_matches_case_name_by_substring() {
        let mut config = PairedConfig::new(3);
        config.filter = Some("high-height".to_owned());
        assert!(config.matches("integer high-height multiplication"));
        assert!(!config.matches("finite-field multiplication"));
    }

    #[test]
    fn csv_fields_are_quoted_and_escape_quotes() {
        assert_eq!(csv_field("large, \"dense\""), "\"large, \"\"dense\"\"\"");
    }

    #[test]
    fn operations_warm_once_and_alternate_sample_order() {
        let log = RefCell::new(Vec::new());
        let symbolica = || {
            log.borrow_mut().push("symbolica");
            DropLog {
                event: "drop-symbolica",
                log: &log,
            }
        };
        let flint = || {
            log.borrow_mut().push("flint");
            DropLog {
                event: "drop-flint",
                log: &log,
            }
        };

        run_paired(&PairedConfig::new(2), "alternating-order", symbolica, flint).unwrap();

        assert_eq!(
            *log.borrow(),
            [
                "symbolica",
                "drop-symbolica",
                "flint",
                "drop-flint",
                "symbolica",
                "drop-symbolica",
                "flint",
                "drop-flint",
                "flint",
                "drop-flint",
                "symbolica",
                "drop-symbolica",
            ]
        );
    }
}
