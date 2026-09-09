//! Reconstruct an ordered set of coefficients using one cached vector trace.
//! Usage: reconstruction_joint_benchmark INPUT_DIR CASE_LIST TRACE SEED
//! CASE_LIST order must match the named outputs of TRACE.
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_void},
    sync::Arc,
    time::Instant,
};
use symbolica::{
    domains::finite_field::{FiniteFieldCore, Zp64},
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod, ReconstructionOptions, reconstruct_rational_function_over_q,
        },
    },
    prelude::*,
};

struct Trace {
    handle: *mut c_void,
    evaluate: unsafe extern "C" fn(*mut c_void, u64, *const u64, usize, *mut u64, usize) -> i32,
    close: unsafe extern "C" fn(*mut c_void),
    count: unsafe extern "C" fn(*mut c_void) -> u64,
    output: Vec<u64>,
    _library: libloading::Library,
}
impl Trace {
    fn open(path: &str, names: &[String], cases: &[String]) -> Self {
        unsafe {
            let library =
                libloading::Library::new(std::env::var("TRACE_ORACLE_LIBRARY").unwrap()).unwrap();
            let open: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> *mut c_void> =
                library.get(b"rr_open_many").unwrap();
            let handle = open(CString::new(path).unwrap().as_ptr());
            assert!(!handle.is_null(), "cannot load joint trace");
            let trace = Self {
                handle,
                evaluate: *library.get(b"rr_evaluate_many").unwrap(),
                close: *library.get(b"rr_close").unwrap(),
                count: *library.get(b"rr_calls").unwrap(),
                output: vec![0; cases.len()],
                _library: library,
            };
            for (count, name, expected) in [
                (b"rr_inputs".as_slice(), b"rr_input_name".as_slice(), names),
                (
                    b"rr_outputs".as_slice(),
                    b"rr_output_name".as_slice(),
                    cases,
                ),
            ] {
                let count: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> usize> =
                    trace._library.get(count).unwrap();
                let name: libloading::Symbol<
                    unsafe extern "C" fn(*mut c_void, usize) -> *const c_char,
                > = trace._library.get(name).unwrap();
                assert_eq!(count(handle), expected.len());
                for (i, expected) in expected.iter().enumerate() {
                    let actual = name(handle, i);
                    assert!(!actual.is_null());
                    assert_eq!(CStr::from_ptr(actual).to_str().unwrap(), expected);
                }
            }
            trace
        }
    }
    fn probe(&mut self, prime: u64, point: &[u64]) -> Option<Vec<u64>> {
        match unsafe {
            (self.evaluate)(
                self.handle,
                prime,
                point.as_ptr(),
                point.len(),
                self.output.as_mut_ptr(),
                self.output.len(),
            )
        } {
            0 => {
                assert!(self.output.iter().all(|x| *x < prime));
                Some(self.output.clone())
            }
            1 => None,
            _ => panic!("trace interpreter error"),
        }
    }
    fn calls(&self) -> u64 {
        unsafe { (self.count)(self.handle) }
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        unsafe { (self.close)(self.handle) };
    }
}

#[derive(Debug)]
struct Limit(&'static str);

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "INPUT_DIR CASE_LIST TRACE SEED");
    let directory = std::path::Path::new(&args[1]);
    let cases: Vec<String> = std::fs::read_to_string(&args[2])
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    assert!(!cases.is_empty());
    let names: Vec<String> =
        std::fs::read_to_string(directory.join(format!("{}.variables", cases[0])))
            .unwrap()
            .split_whitespace()
            .map(String::from)
            .collect();
    let vars: Arc<Vec<PolyVariable>> =
        Arc::new(names.iter().map(|s| symbol!(s.as_str()).into()).collect());
    let originals: Vec<RationalPolynomial<_, u16>> = cases
        .iter()
        .map(|case| {
            let order: Vec<_> =
                std::fs::read_to_string(directory.join(format!("{case}.variables")))
                    .unwrap()
                    .split_whitespace()
                    .map(String::from)
                    .collect();
            assert_eq!(order, names);
            let source = std::fs::read_to_string(directory.join(case)).unwrap();
            parse!(source.trim().trim_end_matches(';')).to_rational_polynomial(
                &Q,
                &Z,
                Some(vars.clone()),
            )
        })
        .collect();
    let seed = args[4].parse().unwrap();
    let options = ReconstructionOptions {
        seed,
        max_degree: 512,
        max_attempts: 2,
        max_probes: 200_000,
        ..Default::default()
    };
    let mut trace = Trace::open(&args[3], &names, &cases);
    // Canonical integers and prime identify a point independently of field representation.
    // Only values are shared; no reference degrees, supports or factors enter the algorithm.
    let mut cache: HashMap<(u64, Vec<u64>), Option<Vec<u64>>> = HashMap::new();
    let cap = std::env::var("MAX_TOTAL_PROBES")
        .map(|s| s.parse().unwrap())
        .unwrap_or(2_000_000u64);
    let timeout = std::env::var("BENCH_TIMEOUT")
        .map(|s| s.parse().unwrap())
        .unwrap_or(570f64);
    let mut requests = 0u64;
    let mut cache_hits = 0u64;
    let max_primes = std::env::var("MAX_PRIMES")
        .map(|s| s.parse().unwrap())
        .unwrap_or(32);
    let mut per_output = Vec::new();
    let mut results = Vec::new();
    let old_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.payload().is::<Limit>() {
            old_hook(info);
        }
    }));
    let start = Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for output in 0..cases.len() {
            let before = trace.calls();
            let result = reconstruct_rational_function_over_q(
                vars.clone(),
                |field: &Zp64, point| {
                    if start.elapsed().as_secs_f64() > timeout {
                        std::panic::panic_any(Limit("time_limit"));
                    }
                    requests += 1;
                    let key = (
                        field.get_prime(),
                        point
                            .iter()
                            .map(|x| field.from_element(x))
                            .collect::<Vec<_>>(),
                    );
                    let value = match cache.entry(key) {
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            cache_hits += 1;
                            entry.into_mut()
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            if trace.calls() >= cap {
                                std::panic::panic_any(Limit("probe_limit"));
                            }
                            let (prime, point) = entry.key();
                            let value = trace.probe(*prime, point);
                            entry.insert(value)
                        }
                    };
                    value
                        .as_ref()
                        .map(|values| field.to_element(values[output]))
                },
                ReconstructionMethod::Automatic,
                &options,
                max_primes,
            );
            per_output.push(trace.calls() - before);
            match result {
                Ok((result, _)) => results.push(result),
                Err(error) => return format!("{error:?}"),
            }
        }
        "ok".to_string()
    }));
    let elapsed = start.elapsed().as_secs_f64() * 1e6;
    let status = match outcome {
        Ok(status) => status,
        Err(e) => match e.downcast::<Limit>() {
            Ok(limit) => limit.0.into(),
            Err(e) => std::panic::resume_unwind(e),
        },
    };
    // Exact validation is outside the timed reconstruction and uses every output.
    for (result, original) in results.iter().zip(&originals) {
        assert_eq!(
            &result.numerator * &original.denominator,
            &result.denominator * &original.numerator
        );
    }
    if status == "ok" {
        assert_eq!(results.len(), originals.len());
    }
    assert_eq!(trace.calls() as usize, cache.len());
    println!(
        "method,seed,status,elapsed_us,probes,scalar_requests,cache_hits,outputs,completed,incremental_probes"
    );
    println!(
        "Symbolica_joint_cache,{seed},{status},{elapsed:.3},{},{requests},{},{},{},{}",
        trace.calls(),
        cache_hits,
        cases.len(),
        results.len(),
        per_output
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(";")
    );
}
