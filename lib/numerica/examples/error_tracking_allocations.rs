//! Count GMP/MPFR allocation traffic during tracked arithmetic.
//!
//! Run with `cargo run -p numerica --release --example error_tracking_allocations -- 200000 current`.
//! This standalone, single-threaded example installs forwarding GMP callbacks
//! before constructing any GMP/MPFR values. Counters cover only each arithmetic
//! loop, after warmup; setup and destruction are excluded. Requested bytes sum
//! allocation sizes and reallocation destination sizes, not peak/live memory.
//! Rust allocations and elapsed time are intentionally not measured.

#[cfg(all(feature = "integer-gmp", feature = "float-mpfr"))]
mod benchmark {
    use std::{
        ffi::c_void,
        hint::black_box,
        sync::{
            OnceLock,
            atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
        },
    };

    use gmp_mpfr_sys::gmp;
    use numerica::domains::float::{Complex, ErrorPropagatingFloat, Float};

    type Tracked = ErrorPropagatingFloat<Float>;

    #[derive(Debug)]
    struct Callbacks {
        allocate: extern "C" fn(usize) -> *mut c_void,
        reallocate: unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void,
        free: unsafe extern "C" fn(*mut c_void, usize),
    }

    static ORIGINAL: OnceLock<Callbacks> = OnceLock::new();
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    static FREES: AtomicU64 = AtomicU64::new(0);
    static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
    static REALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

    extern "C" fn allocate(size: usize) -> *mut c_void {
        if ENABLED.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED_BYTES.fetch_add(size as u64, Relaxed);
        }
        // The original callbacks are recorded before these hooks are installed.
        (ORIGINAL.get().unwrap().allocate)(size)
    }

    unsafe extern "C" fn reallocate(
        pointer: *mut c_void,
        old_size: usize,
        new_size: usize,
    ) -> *mut c_void {
        if ENABLED.load(Relaxed) {
            REALLOCATIONS.fetch_add(1, Relaxed);
            REALLOCATED_BYTES.fetch_add(new_size as u64, Relaxed);
        }
        unsafe { (ORIGINAL.get().unwrap().reallocate)(pointer, old_size, new_size) }
    }

    unsafe extern "C" fn free(pointer: *mut c_void, size: usize) {
        if ENABLED.load(Relaxed) {
            FREES.fetch_add(1, Relaxed);
        }
        unsafe { (ORIGINAL.get().unwrap().free)(pointer, size) }
    }

    fn install_hooks() {
        let (mut allocate_original, mut reallocate_original, mut free_original) =
            (None, None, None);
        // This is called once at process startup, before any GMP-backed value
        // or worker thread exists. Every callback forwards to the same allocator.
        unsafe {
            gmp::get_memory_functions(
                &mut allocate_original,
                &mut reallocate_original,
                &mut free_original,
            );
            ORIGINAL
                .set(Callbacks {
                    allocate: allocate_original.unwrap(),
                    reallocate: reallocate_original.unwrap(),
                    free: free_original.unwrap(),
                })
                .unwrap();
            gmp::set_memory_functions(Some(allocate), Some(reallocate), Some(free));
        }
    }

    fn measure<T>(
        revision: &str,
        name: &str,
        iterations: usize,
        value: &mut T,
        rhs: &T,
        mut operation: impl FnMut(&mut T, &T),
    ) {
        for _ in 0..32 {
            operation(black_box(&mut *value), black_box(rhs));
        }
        for counter in [
            &ALLOCATIONS,
            &REALLOCATIONS,
            &FREES,
            &ALLOCATED_BYTES,
            &REALLOCATED_BYTES,
        ] {
            counter.store(0, Relaxed);
        }
        ENABLED.store(true, Relaxed);
        for _ in 0..iterations {
            operation(black_box(&mut *value), black_box(rhs));
        }
        ENABLED.store(false, Relaxed);
        black_box(&*value);

        let allocations = ALLOCATIONS.load(Relaxed);
        let reallocations = REALLOCATIONS.load(Relaxed);
        let frees = FREES.load(Relaxed);
        let allocated_bytes = ALLOCATED_BYTES.load(Relaxed);
        let reallocated_bytes = REALLOCATED_BYTES.load(Relaxed);
        println!(
            "{revision},{name},{iterations},{allocations},{reallocations},{frees},{allocated_bytes},{reallocated_bytes},{:.3},{:.3}",
            (allocations + reallocations) as f64 / iterations as f64,
            (allocated_bytes + reallocated_bytes) as f64 / iterations as f64,
        );
    }

    fn scalar(value: f64) -> Tracked {
        ErrorPropagatingFloat::new(Float::with_val(256, value), 65.)
    }

    pub fn run() {
        install_hooks();
        let mut arguments = std::env::args().skip(1);
        let iterations = arguments
            .next()
            .map(|value| value.parse::<usize>().expect("invalid iteration count"))
            .unwrap_or(200_000);
        assert!(iterations > 0, "iteration count must be positive");
        let revision = arguments.next().unwrap_or_else(|| "current".to_owned());
        println!(
            "revision,operation,iterations,allocations,reallocations,frees,allocated_bytes,reallocated_bytes,allocation_requests_per_op,requested_bytes_per_op"
        );

        let rhs = scalar(1.000_001);
        let mut value = scalar(1.);
        measure(
            &revision,
            "tracked mul_assign",
            iterations,
            &mut value,
            &rhs,
            |a, b| *a *= b,
        );
        assert!(value.get_num().is_finite());
        let mut value = scalar(1.);
        measure(
            &revision,
            "tracked div_assign",
            iterations,
            &mut value,
            &rhs,
            |a, b| *a /= b,
        );
        assert!(value.get_num().is_finite());

        // Exact imaginary zeros select the purely real complex paths; an
        // uncertain zero would deliberately exercise different operations.
        let rhs = Complex::new(scalar(1.000_001), Tracked::exact_zero(256));
        let mut value = Complex::new(scalar(1.), Tracked::exact_zero(256));
        measure(
            &revision,
            "exact-real complex mul_assign",
            iterations,
            &mut value,
            &rhs,
            |a, b| *a *= b,
        );
        assert!(value.re.get_num().is_finite() && value.im.is_exact_zero());
        let mut value = Complex::new(scalar(1.), Tracked::exact_zero(256));
        measure(
            &revision,
            "exact-real complex div_assign",
            iterations,
            &mut value,
            &rhs,
            |a, b| *a /= b,
        );
        assert!(value.re.get_num().is_finite() && value.im.is_exact_zero());
    }
}

#[cfg(all(feature = "integer-gmp", feature = "float-mpfr"))]
fn main() {
    benchmark::run();
}

#[cfg(not(all(feature = "integer-gmp", feature = "float-mpfr")))]
fn main() {
    eprintln!("This allocation benchmark requires the integer-gmp and float-mpfr features.");
}
