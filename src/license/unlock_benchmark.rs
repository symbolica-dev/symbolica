//! An opt-in microbenchmark of library-unlock bookkeeping, without license keys or I/O.
//!
//! Run in release mode; see `benches/library_unlock.md` for the measurement protocol.

use std::{hint::black_box, time::Instant};

use super::*;

const SAMPLES: usize = 11;
const CHECK_ITERATIONS: usize = 10_000_000;
const ACTIVATION_ITERATIONS: usize = 500_000;

// Preserve the implementation from 1a1829b1 here as the comparison baseline. The old
// inherited token was a Boolean snapshot, and each activation allocated a fresh Arc.
// Keep its TLS separate so it cannot accidentally authorize the implementation under test.
mod legacy {
    use super::*;

    std::thread_local! {
        static UNLOCKS: RefCell<Vec<Arc<AtomicBool>>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) struct Guard {
        enabled: Arc<AtomicBool>,
        _not_send: PhantomData<Rc<()>>,
    }

    impl Guard {
        #[inline]
        pub(super) fn with_license(enabled: Arc<AtomicBool>) -> Self {
            UNLOCKS.with(|licenses| licenses.borrow_mut().push(enabled.clone()));
            Self {
                enabled,
                _not_send: PhantomData,
            }
        }
    }

    impl Drop for Guard {
        #[inline]
        fn drop(&mut self) {
            UNLOCKS.with(|licenses| {
                let mut licenses = licenses.borrow_mut();
                let index = licenses
                    .iter()
                    .rposition(|license| Arc::ptr_eq(license, &self.enabled))
                    .expect("unbalanced library unlock scope");
                licenses.remove(index);
            });
        }
    }

    #[inline]
    pub(super) fn current_thread_is_unlocked() -> bool {
        UNLOCKS.with(|licenses| {
            licenses
                .borrow()
                .iter()
                .any(|license| license.load(Relaxed))
        })
    }

    #[derive(Clone, Copy)]
    pub(super) struct Inherited(bool);

    impl Inherited {
        #[inline]
        pub(super) fn capture() -> Self {
            Self(current_thread_is_unlocked())
        }

        #[inline]
        pub(super) fn activate(self) -> Option<Guard> {
            self.0
                .then(|| Guard::with_license(Arc::new(AtomicBool::new(true))))
        }
    }
}

fn measure(iterations: usize, mut operation: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

fn compare(
    name: &str,
    iterations: usize,
    mut legacy_sample: impl FnMut(usize) -> f64,
    mut scoped_sample: impl FnMut(usize) -> f64,
) {
    // Warm TLS vectors and allocation caches outside the reported measurements.
    legacy_sample(10_000);
    scoped_sample(10_000);

    let mut legacy = [0.0_f64; SAMPLES];
    let mut scoped = [0.0_f64; SAMPLES];
    for sample in 0..SAMPLES {
        // Alternate order to reduce systematic bias from clock/temperature drift.
        if sample % 2 == 0 {
            legacy[sample] = legacy_sample(iterations);
            scoped[sample] = scoped_sample(iterations);
        } else {
            scoped[sample] = scoped_sample(iterations);
            legacy[sample] = legacy_sample(iterations);
        }
    }
    legacy.sort_by(f64::total_cmp);
    scoped.sort_by(f64::total_cmp);
    println!(
        "{name},{iterations},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        legacy[SAMPLES / 2],
        scoped[SAMPLES / 2],
        scoped[SAMPLES / 2] / legacy[SAMPLES / 2],
        legacy[0],
        legacy[SAMPLES - 1],
        scoped[0],
        scoped[SAMPLES - 1],
    );
}

#[test]
#[ignore = "microbenchmark: run explicitly in release mode on an otherwise idle machine"]
fn scoped_library_unlock_overhead() {
    assert!(
        !cfg!(debug_assertions),
        "this microbenchmark requires cargo test --release"
    );
    assert!(!current_thread_is_unlocked());
    let enabled = Arc::new(AtomicBool::new(black_box(true)));
    let registration = LibraryUnlock {
        enabled: enabled.clone(),
    };
    let _source = registration.unlock();
    let _legacy_source = legacy::Guard::with_license(enabled);
    assert!(current_thread_is_unlocked());
    assert!(legacy::current_thread_is_unlocked());

    println!(
        "case,iterations,legacy_median_ns,scoped_median_ns,ratio,legacy_min_ns,legacy_max_ns,scoped_min_ns,scoped_max_ns"
    );
    compare(
        "direct_hot_check",
        CHECK_ITERATIONS,
        |iterations| {
            measure(iterations, || {
                black_box(legacy::current_thread_is_unlocked());
            })
        },
        |iterations| {
            measure(iterations, || {
                black_box(current_thread_is_unlocked());
            })
        },
    );

    compare(
        "capture_and_exit",
        ACTIVATION_ITERATIONS,
        |iterations| {
            measure(iterations, || {
                black_box(legacy::Inherited::capture());
            })
        },
        |iterations| {
            measure(iterations, || {
                LicenseManager::with_inherited_library_unlock(|scope| {
                    black_box(scope);
                });
            })
        },
    );

    let legacy_inherited = legacy::Inherited::capture();
    LicenseManager::with_inherited_library_unlock(|scope| {
        std::thread::scope(|threads| {
            threads
                .spawn(|| {
                    // A separate worker is essential: a direct caller-side guard would
                    // otherwise satisfy the check before it reaches inherited state.
                    assert!(!current_thread_is_unlocked());
                    assert!(!legacy::current_thread_is_unlocked());
                    compare(
                        "inherited_hot_check",
                        CHECK_ITERATIONS,
                        |iterations| {
                            let _guard = legacy_inherited.activate();
                            assert!(legacy::current_thread_is_unlocked());
                            measure(iterations, || {
                                black_box(legacy::current_thread_is_unlocked());
                            })
                        },
                        |iterations| {
                            scope.run(|| {
                                assert!(current_thread_is_unlocked());
                                measure(iterations, || {
                                    black_box(current_thread_is_unlocked());
                                })
                            })
                        },
                    );
                    compare(
                        "worker_activation_check_drop",
                        ACTIVATION_ITERATIONS,
                        |iterations| {
                            measure(iterations, || {
                                let _guard = black_box(legacy_inherited).activate();
                                black_box(legacy::current_thread_is_unlocked());
                            })
                        },
                        |iterations| {
                            measure(iterations, || {
                                black_box(&scope).run(|| {
                                    black_box(current_thread_is_unlocked());
                                });
                            })
                        },
                    );

                    scope.run(|| {
                        LicenseManager::with_inherited_library_unlock(|nested| {
                            std::thread::scope(|children| {
                                children
                                    .spawn(|| {
                                        assert!(!current_thread_is_unlocked());
                                        compare(
                                            "nested_inherited_hot_check",
                                            CHECK_ITERATIONS,
                                            |iterations| {
                                                // The old representation discarded ancestry:
                                                // a nested capture remained the same bool.
                                                let _guard = legacy_inherited.activate();
                                                assert!(legacy::current_thread_is_unlocked());
                                                measure(iterations, || {
                                                    black_box(legacy::current_thread_is_unlocked());
                                                })
                                            },
                                            |iterations| {
                                                nested.run(|| {
                                                    assert!(current_thread_is_unlocked());
                                                    measure(iterations, || {
                                                        black_box(current_thread_is_unlocked());
                                                    })
                                                })
                                            },
                                        );
                                    })
                                    .join()
                                    .unwrap();
                            });
                        });
                    });
                    assert!(!current_thread_is_unlocked());
                    assert!(!legacy::current_thread_is_unlocked());
                })
                .join()
                .unwrap();
        });
    });

    // Two independently authorized libraries exercise the branching provenance path.
    let second_enabled = Arc::new(AtomicBool::new(black_box(true)));
    let second_registration = LibraryUnlock {
        enabled: second_enabled.clone(),
    };
    let _second_source = second_registration.unlock();
    let _second_legacy_source = legacy::Guard::with_license(second_enabled);
    let legacy_inherited = legacy::Inherited::capture();
    LicenseManager::with_inherited_library_unlock(|scope| {
        std::thread::scope(|threads| {
            threads
                .spawn(|| {
                    assert!(!current_thread_is_unlocked());
                    compare(
                        "two_source_inherited_hot_check",
                        CHECK_ITERATIONS,
                        |iterations| {
                            let _guard = legacy_inherited.activate();
                            assert!(legacy::current_thread_is_unlocked());
                            measure(iterations, || {
                                black_box(legacy::current_thread_is_unlocked());
                            })
                        },
                        |iterations| {
                            scope.run(|| {
                                assert!(current_thread_is_unlocked());
                                measure(iterations, || {
                                    black_box(current_thread_is_unlocked());
                                })
                            })
                        },
                    );
                })
                .join()
                .unwrap();
        });
    });
}

const CONTENDING_WORKERS: usize = 4;

fn measure_contending_workers(iterations: usize, operation: impl Fn() + Sync) -> f64 {
    use std::sync::Barrier;

    let ready = Barrier::new(CONTENDING_WORKERS + 1);
    let start = Barrier::new(CONTENDING_WORKERS + 1);
    let finished = Barrier::new(CONTENDING_WORKERS + 1);
    std::thread::scope(|threads| {
        for _ in 0..CONTENDING_WORKERS {
            let (ready, start, finished, operation) = (&ready, &start, &finished, &operation);
            threads.spawn(move || {
                assert!(!current_thread_is_unlocked());
                assert!(!legacy::current_thread_is_unlocked());
                ready.wait();
                start.wait();
                for _ in 0..iterations {
                    operation();
                }
                finished.wait();
                assert!(!current_thread_is_unlocked());
                assert!(!legacy::current_thread_is_unlocked());
            });
        }
        ready.wait();
        let timer = Instant::now();
        start.wait();
        finished.wait();
        // Fixed start/finish barrier costs are included; thread creation and joins are not.
        timer.elapsed().as_secs_f64() * 1e9 / (CONTENDING_WORKERS * iterations) as f64
    })
}

#[test]
#[ignore = "microbenchmark: run in release mode with at least four physical cores available"]
fn scoped_library_unlock_contended_activation() {
    assert!(
        !cfg!(debug_assertions),
        "this microbenchmark requires cargo test --release"
    );
    let enabled = Arc::new(AtomicBool::new(black_box(true)));
    let registration = LibraryUnlock {
        enabled: enabled.clone(),
    };
    let _source = registration.unlock();
    let _legacy_source = legacy::Guard::with_license(enabled);
    let legacy_inherited = legacy::Inherited::capture();
    println!(
        "Four concurrent workers; times are wall-clock ns per aggregate activation, not single-worker latency."
    );
    println!(
        "case,iterations_per_worker,legacy_median_ns,scoped_median_ns,ratio,legacy_min_ns,legacy_max_ns,scoped_min_ns,scoped_max_ns"
    );
    LicenseManager::with_inherited_library_unlock(|scope| {
        for (checks, iterations) in [(1, 250_000), (1_000, 2_000)] {
            compare(
                &format!("four_workers_activation_{checks}_checks"),
                iterations,
                |iterations| {
                    measure_contending_workers(iterations, || {
                        let _guard = black_box(legacy_inherited).activate();
                        for _ in 0..checks {
                            black_box(legacy::current_thread_is_unlocked());
                        }
                    })
                },
                |iterations| {
                    measure_contending_workers(iterations, || {
                        black_box(&scope).run(|| {
                            for _ in 0..checks {
                                black_box(current_thread_is_unlocked());
                            }
                        });
                    })
                },
            );
        }
    });
}
