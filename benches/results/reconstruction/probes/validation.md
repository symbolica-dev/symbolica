Validation uses the locked dependencies and build settings of commit `306c375`.

- `cargo test --locked --test rational_reconstruction --test rational_polynomial`: 20 tests passed; see `tests.log`.
- The probe-budget test covers ten seeds at each of the 509- and 306-probe limits, with independent exact identities and actual callback counts.
- Generated sparse and special-case tests cover all three methods. Nonseparable denominator fallback is exercised with one allowed attempt.
- Rust formatting checks and `git diff --check` passed.
- Clippy passes with the existing `clippy::never_loop` diagnostic allowed; 248 pre-existing library warnings, no diagnostics in reconstruction code or the three benchmark examples.
- Every recorded benchmark run independently checks the exact polynomial cross-product outside its timer.
