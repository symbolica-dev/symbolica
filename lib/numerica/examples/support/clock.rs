//! Timing for the single-threaded arithmetic benchmark loops.
//!
//! Linux with a 64-bit ABI defaults to thread CPU time, excluding time spent
//! descheduled. Set NUMERICA_BENCH_CLOCK=wall to use Instant instead. CPU time
//! still varies with CPU frequency, cache contention, and migration between cores.

use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Clock {
    Wall,
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    ThreadCpu,
}

impl Clock {
    pub fn from_env() -> Self {
        let requested = std::env::var("NUMERICA_BENCH_CLOCK").unwrap_or_else(|_| "auto".into());
        assert!(
            matches!(requested.as_str(), "auto" | "thread" | "wall"),
            "NUMERICA_BENCH_CLOCK must be auto, thread, or wall"
        );
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        if requested != "wall" && linux::thread_cpu_time().is_some() {
            return Self::ThreadCpu;
        }
        if requested == "thread" {
            eprintln!("Thread CPU clock unavailable; falling back to Instant wall time");
        }
        Self::Wall
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Wall => "Instant wall time (includes scheduler delays)",
            #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
            Self::ThreadCpu => "Linux CLOCK_THREAD_CPUTIME_ID (excludes scheduler delays)",
        }
    }

    pub fn start(self) -> Timer {
        match self {
            Self::Wall => Timer::Wall(Instant::now()),
            #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
            Self::ThreadCpu => Timer::ThreadCpu(
                linux::thread_cpu_time().expect("clock_gettime failed when starting measurement"),
            ),
        }
    }
}

pub enum Timer {
    Wall(Instant),
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    ThreadCpu(Duration),
}

impl Timer {
    /// Read on the same thread that started this measurement.
    pub fn elapsed(&self) -> Duration {
        match self {
            Self::Wall(start) => start.elapsed(),
            #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
            Self::ThreadCpu(start) => linux::thread_cpu_time()
                .expect("clock_gettime failed during measurement")
                .checked_sub(*start)
                .expect("thread CPU clock moved backwards"),
        }
    }
}

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
mod linux {
    use std::{
        ffi::{c_int, c_long},
        mem::MaybeUninit,
        time::Duration,
    };

    // Linux's 64-bit ABI uses signed 64-bit long values for both fields.
    #[repr(C)]
    struct Timespec {
        tv_sec: c_long,
        tv_nsec: c_long,
    }

    #[link(name = "c")]
    unsafe extern "C" {
        fn clock_gettime(clock_id: c_int, timestamp: *mut Timespec) -> c_int;
    }

    pub(super) fn thread_cpu_time() -> Option<Duration> {
        const CLOCK_THREAD_CPUTIME_ID: c_int = 3;
        let mut timestamp = MaybeUninit::<Timespec>::uninit();
        // SAFETY: the Linux ABI and clock ID match the declaration above; the
        // pointer refers to sufficient, writable storage for one timespec.
        if unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, timestamp.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: successful clock_gettime initialized both timespec fields.
        let timestamp = unsafe { timestamp.assume_init() };
        if timestamp.tv_sec < 0 || !(0..1_000_000_000).contains(&timestamp.tv_nsec) {
            return None;
        }
        Some(Duration::new(
            timestamp.tv_sec as u64,
            timestamp.tv_nsec as u32,
        ))
    }
}
