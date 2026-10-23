//! Cooperative process-memory and elapsed-time budgets shared by nested work.
//!
//! No work-count limits are imposed. Call [`checkpoint`] at interruptible
//! boundaries and use [`ResourceBudget::capture_current`] when spawning workers.
//! Opaque external calls cannot be interrupted here; a process supervisor is
//! required for a hard deadline. RSS covers the whole process, not an individual
//! expression, cache, or thread. Concurrent allocations can exceed an observed
//! headroom estimate before the next checkpoint.

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

/// Explicit resource ceilings; an absent ceiling imposes no restriction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct ResourceLimits {
    pub max_process_rss_bytes: Option<u64>,
    pub max_elapsed: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceLimitError {
    Memory {
        rss_bytes: u64,
        additional_bytes: u64,
        limit_bytes: u64,
    },
    Elapsed {
        elapsed: Duration,
        limit: Duration,
    },
    MonitorUnavailable {
        resource: &'static str,
        message: String,
    },
}

impl fmt::Display for ResourceLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory {
                rss_bytes,
                additional_bytes,
                limit_bytes,
            } => write!(
                f,
                "process RSS {rss_bytes} bytes plus requested allocation {additional_bytes} bytes exceeds RAM budget {limit_bytes} bytes"
            ),
            Self::Elapsed { elapsed, limit } => write!(
                f,
                "elapsed time {elapsed:?} reached the configured time budget {limit:?}"
            ),
            Self::MonitorUnavailable { resource, message } => {
                write!(f, "cannot enforce {resource} budget: {message}")
            }
        }
    }
}

impl std::error::Error for ResourceLimitError {}

type MemoryMonitor = dyn Fn() -> Result<u64, String> + Send + Sync;

enum Budget {
    Limited {
        limits: ResourceLimits,
        started: Option<Instant>,
        memory: Arc<MemoryMonitor>,
    },
    Captured(Vec<ResourceBudget>),
}

/// Clones share the same clock and monitor; they do not start another allowance.
#[derive(Clone)]
pub struct ResourceBudget(Arc<Budget>);

impl Default for ResourceBudget {
    fn default() -> Self {
        Self::new(ResourceLimits::default())
    }
}

impl fmt::Debug for ResourceBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_ref() {
            Budget::Limited {
                limits, started, ..
            } => f
                .debug_struct("ResourceBudget")
                .field("limits", limits)
                .field("elapsed", &started.as_ref().map(Instant::elapsed))
                .finish_non_exhaustive(),
            Budget::Captured(budgets) => f
                .debug_tuple("CapturedResourceBudgets")
                .field(budgets)
                .finish(),
        }
    }
}

thread_local! {
    static ACTIVE: RefCell<Vec<ResourceBudget>> = const { RefCell::new(Vec::new()) };
}

impl ResourceBudget {
    /// Linux supplies process RSS through `/proc/self/status`. Other platforms
    /// need [`Self::with_memory_monitor`] when an RSS ceiling is requested.
    pub fn new(limits: ResourceLimits) -> Self {
        Self::with_memory_monitor(limits, process_rss_bytes)
    }

    /// The monitor reports bytes for the same process-memory domain as the
    /// ceiling. It must not enter another budget scope or call checkpoints.
    pub fn with_memory_monitor(
        limits: ResourceLimits,
        monitor: impl Fn() -> Result<u64, String> + Send + Sync + 'static,
    ) -> Self {
        // std::time::Instant is unavailable on wasm32-unknown-unknown. Unbounded
        // budgets still work there; a requested unsupported clock fails closed.
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        let started = limits.max_elapsed.map(|_| Instant::now());
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let started = None;
        Self(Arc::new(Budget::Limited {
            limits,
            started,
            memory: Arc::new(monitor),
        }))
    }

    /// Install this budget for a lexical operation. Checkpoints are explicit;
    /// nesting never replaces an outer restriction. Panic unwinding restores
    /// the previous context. Spawned threads require [`Self::capture_current`].
    pub fn run<R>(&self, operation: impl FnOnce() -> R) -> R {
        if self.is_unlimited() {
            return operation();
        }
        ACTIVE.with(|active| active.borrow_mut().push(self.clone()));
        struct Pop;
        impl Drop for Pop {
            fn drop(&mut self) {
                ACTIVE.with(|active| {
                    active.borrow_mut().pop();
                });
            }
        }
        let _scope = Pop;
        operation()
    }

    /// Capture every active restriction for reuse on another thread. The clock
    /// origins remain unchanged, including while the worker waits to execute.
    pub fn capture_current() -> Self {
        ACTIVE.with(|active| {
            let mut budgets = Vec::new();
            let mut seen = HashSet::new();
            let mut insert = |budget: &Self| {
                if seen.insert(Arc::as_ptr(&budget.0)) {
                    budgets.push(budget.clone());
                }
            };
            for budget in active.borrow().iter() {
                match budget.0.as_ref() {
                    Budget::Limited { .. } => insert(budget),
                    // Captured contexts contain only individual budgets. Keep
                    // captures flat to avoid recursive/exponential context growth.
                    Budget::Captured(items) => items.iter().for_each(&mut insert),
                }
            }
            Self(Arc::new(Budget::Captured(budgets)))
        })
    }

    /// Check this budget or captured context. Use the free [`checkpoint`]
    /// function to check every budget active on the calling thread.
    pub fn checkpoint(&self) -> Result<(), ResourceLimitError> {
        self.check_allocation(0)
    }

    /// Check this budget's RSS and anticipated additional bytes before allocation.
    /// This is a cooperative headroom check, not an allocator reservation.
    /// Use the free [`check_allocation`] function for all active scopes.
    pub fn check_allocation(&self, additional_bytes: u64) -> Result<(), ResourceLimitError> {
        match self.0.as_ref() {
            Budget::Captured(budgets) => {
                for budget in budgets {
                    budget.check_allocation(additional_bytes)?;
                }
            }
            Budget::Limited {
                limits,
                started,
                memory,
            } => {
                if let Some(limit) = limits.max_elapsed {
                    let elapsed = started
                        .ok_or_else(|| ResourceLimitError::MonitorUnavailable {
                            resource: "elapsed time",
                            message: "std::time::Instant is unavailable on this target".into(),
                        })?
                        .elapsed();
                    if elapsed >= limit {
                        return Err(ResourceLimitError::Elapsed { elapsed, limit });
                    }
                }
                if let Some(limit_bytes) = limits.max_process_rss_bytes {
                    let rss_bytes =
                        memory().map_err(|message| ResourceLimitError::MonitorUnavailable {
                            resource: "process RSS",
                            message,
                        })?;
                    if u128::from(rss_bytes) + u128::from(additional_bytes)
                        > u128::from(limit_bytes)
                    {
                        return Err(ResourceLimitError::Memory {
                            rss_bytes,
                            additional_bytes,
                            limit_bytes,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn is_unlimited(&self) -> bool {
        match self.0.as_ref() {
            Budget::Limited { limits, .. } => {
                limits.max_elapsed.is_none() && limits.max_process_rss_bytes.is_none()
            }
            Budget::Captured(budgets) => budgets.iter().all(Self::is_unlimited),
        }
    }
}

/// Check all active scopes. With no configured budgets this performs no memory
/// query or clock read. A nested unbounded operation keeps its outer budget.
pub fn checkpoint() -> Result<(), ResourceLimitError> {
    check_allocation(0)
}

/// Check anticipated allocation headroom against every active scope.
pub fn check_allocation(additional_bytes: u64) -> Result<(), ResourceLimitError> {
    ACTIVE.with(|active| {
        for budget in active.borrow().iter() {
            budget.check_allocation(additional_bytes)?;
        }
        Ok(())
    })
}

#[cfg(target_os = "linux")]
fn process_rss_bytes() -> Result<u64, String> {
    let status = std::fs::read_to_string("/proc/self/status").map_err(|error| error.to_string())?;
    parse_rss(&status)
}

#[cfg(target_os = "linux")]
fn parse_rss(status: &str) -> Result<u64, String> {
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .ok_or_else(|| "VmRSS is absent from /proc/self/status".to_owned())?;
    let mut fields = line.split_whitespace();
    let kib: u64 = fields
        .next()
        .ok_or_else(|| "VmRSS value is absent".to_owned())?
        .parse()
        .map_err(|error| format!("invalid VmRSS value: {error}"))?;
    if fields.next() != Some("kB") || fields.next().is_some() {
        return Err("unexpected VmRSS unit or trailing data".into());
    }
    kib.checked_mul(1024)
        .ok_or_else(|| "VmRSS byte count cannot be represented by u64".into())
}

#[cfg(not(target_os = "linux"))]
fn process_rss_bytes() -> Result<u64, String> {
    Err("supply a process-memory monitor for this platform".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(limit: u64, observed: u64) -> ResourceBudget {
        ResourceBudget::with_memory_monitor(
            ResourceLimits {
                max_process_rss_bytes: Some(limit),
                ..Default::default()
            },
            move || Ok(observed),
        )
    }

    #[test]
    fn default_does_not_query_resources_or_restrict_work() {
        let budget = ResourceBudget::with_memory_monitor(ResourceLimits::default(), || {
            panic!("unrequested memory query")
        });
        budget.run(checkpoint).unwrap();
        budget.check_allocation(u64::MAX).unwrap();
    }

    #[test]
    fn memory_headroom_includes_pending_allocation_without_overflow() {
        let budget = memory(100, 80);
        budget.check_allocation(20).unwrap();
        assert_eq!(
            budget.check_allocation(21),
            Err(ResourceLimitError::Memory {
                rss_bytes: 80,
                additional_bytes: 21,
                limit_bytes: 100,
            })
        );
        assert!(memory(u64::MAX, u64::MAX).check_allocation(1).is_err());
    }

    #[test]
    fn missing_requested_monitor_fails_explicitly() {
        let budget = ResourceBudget::with_memory_monitor(
            ResourceLimits {
                max_process_rss_bytes: Some(100),
                ..Default::default()
            },
            || Err("monitor disconnected".into()),
        );
        assert_eq!(
            budget.checkpoint(),
            Err(ResourceLimitError::MonitorUnavailable {
                resource: "process RSS",
                message: "monitor disconnected".into(),
            })
        );
    }

    #[test]
    fn zero_time_expires_and_worker_clones_do_not_restart_the_deadline() {
        let budget = ResourceBudget::new(ResourceLimits {
            max_elapsed: Some(Duration::ZERO),
            ..Default::default()
        });
        assert!(matches!(
            budget.checkpoint(),
            Err(ResourceLimitError::Elapsed { .. })
        ));
        let budget = ResourceBudget::new(ResourceLimits {
            max_elapsed: Some(Duration::from_millis(2)),
            ..Default::default()
        });
        std::thread::sleep(Duration::from_millis(5));
        let cloned = budget.clone();
        assert!(matches!(
            std::thread::spawn(move || cloned.checkpoint())
                .join()
                .unwrap(),
            Err(ResourceLimitError::Elapsed { .. })
        ));
    }

    #[test]
    fn nested_unbounded_work_cannot_relax_an_outer_budget() {
        let result =
            memory(5, 6).run(|| ResourceBudget::new(ResourceLimits::default()).run(checkpoint));
        assert!(matches!(result, Err(ResourceLimitError::Memory { .. })));
        checkpoint().unwrap();
    }

    #[test]
    fn captured_context_keeps_all_scopes_on_a_worker() {
        let captured = memory(5, 6).run(|| memory(100, 1).run(ResourceBudget::capture_current));
        let worker = std::thread::spawn(move || captured.run(checkpoint));
        assert!(matches!(
            worker.join().unwrap(),
            Err(ResourceLimitError::Memory { .. })
        ));
        checkpoint().unwrap();
    }

    #[test]
    fn nested_allocation_checks_and_repeated_captures_keep_outer_limits() {
        memory(10, 8).run(|| {
            assert!(check_allocation(3).is_err());
            let mut captured = ResourceBudget::capture_current();
            for _ in 0..100 {
                captured = captured.run(ResourceBudget::capture_current);
            }
            let Budget::Captured(items) = captured.0.as_ref() else {
                panic!("expected capture")
            };
            assert_eq!(items.len(), 1);
            assert!(captured.check_allocation(3).is_err());
        });
    }

    #[test]
    fn panic_unwinding_restores_the_outer_context() {
        let budget = memory(5, 6);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            budget.run(|| panic!("failed operation"))
        }));
        assert!(result.is_err());
        checkpoint().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_monitor_uses_current_rss_and_rejects_malformed_reports() {
        assert_eq!(parse_rss("Name:\ttest\nVmRSS:\t123 kB\n"), Ok(123 * 1024));
        assert!(process_rss_bytes().unwrap() > 0);
        for invalid in [
            "",
            "VmRSS: 1 MB",
            "VmRSS: 1 kB extra",
            "VmRSS: -1 kB",
            "VmRSS: 18446744073709551615 kB",
        ] {
            assert!(parse_rss(invalid).is_err());
        }
    }
}
