//! Owned wrappers for the FLINT polynomial values used by the benchmarks.
//!
//! Each operation creates a new FLINT output polynomial. Input construction,
//! parsing, and result validation can therefore stay outside the timed region,
//! while a timed operation includes allocation of its returned value just like
//! the corresponding Symbolica operation.

use flint3_sys as ffi;
use std::ffi::{CStr, CString, c_char};
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Once;

static INITIALIZE_FLINT: Once = Once::new();

unsafe extern "C" {
    #[link_name = "flint_version"]
    static FLINT_VERSION_STRING: c_char;
}

/// Selects the FLINT GCD implementation measured by a benchmark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GcdAlgorithm {
    /// Let FLINT select a GCD implementation.
    Auto,
    /// Use FLINT's Hensel-lifting GCD implementation.
    Hensel,
    /// Use FLINT's Zippel GCD implementation.
    Zippel,
    /// Use FLINT's Zippel2 GCD implementation.
    Zippel2,
}

/// An error reported while constructing a FLINT value or running an operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlintError(String);

impl FlintError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for FlintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FlintError {}

/// Configures FLINT's global worker pool for single-core measurements.
///
/// Call this before constructing benchmark inputs. Repeated calls are harmless.
pub fn initialize_single_thread() {
    INITIALIZE_FLINT.call_once(|| {
        // SAFETY: The call is serialized by `INITIALIZE_FLINT` and happens before
        // the benchmark creates values or enters timed operations.
        unsafe { ffi::flint_set_num_threads(1) };
    });
}

/// Returns the version string exported by the linked FLINT library.
pub fn version() -> String {
    // SAFETY: FLINT exports `flint_version` as a process-lifetime,
    // null-terminated byte array.
    unsafe {
        CStr::from_ptr(&FLINT_VERSION_STRING)
            .to_string_lossy()
            .into_owned()
    }
}

fn encode_variable_names<S: AsRef<str>>(
    variable_names: &[S],
) -> Result<(Vec<CString>, Vec<*const c_char>), FlintError> {
    if variable_names.is_empty() {
        return Err(FlintError::new(
            "a FLINT multivariate polynomial context needs at least one variable",
        ));
    }
    if variable_names.len() > ffi::slong::MAX as usize {
        return Err(FlintError::new("too many variables for a FLINT context"));
    }

    let names = variable_names
        .iter()
        .map(|name| {
            CString::new(name.as_ref()).map_err(|_| {
                FlintError::new(format!(
                    "variable name {:?} contains a null byte",
                    name.as_ref()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pointers = names.iter().map(|name| name.as_ptr()).collect();
    Ok((names, pointers))
}

fn encode_expression(expression: &str) -> Result<CString, FlintError> {
    CString::new(expression)
        .map_err(|_| FlintError::new("polynomial expression contains a null byte"))
}

/// A lexicographically ordered FLINT multivariate polynomial context over the integers.
pub struct FmpzMPolyContext {
    raw: Box<ffi::fmpz_mpoly_ctx_struct>,
    _variable_names: Vec<CString>,
    variable_pointers: Vec<*const c_char>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl FmpzMPolyContext {
    /// Creates an integer polynomial context with the supplied variable order.
    pub fn new<S: AsRef<str>>(variable_names: &[S]) -> Result<Self, FlintError> {
        initialize_single_thread();
        let (variable_names, variable_pointers) = encode_variable_names(variable_names)?;
        let mut raw = Box::<ffi::fmpz_mpoly_ctx_struct>::new_uninit();

        // SAFETY: `raw` points to suitably aligned, writable storage, the variable
        // count fits `slong`, and FLINT fully initializes the context.
        unsafe {
            ffi::fmpz_mpoly_ctx_init(
                raw.as_mut_ptr(),
                variable_names.len() as ffi::slong,
                ffi::ordering_t_ORD_LEX,
            );
        }

        Ok(Self {
            // SAFETY: `fmpz_mpoly_ctx_init` completed initialization above.
            raw: unsafe { raw.assume_init() },
            _variable_names: variable_names,
            variable_pointers,
            _not_send_or_sync: PhantomData,
        })
    }

    /// Parses an integer polynomial using this context's variable names.
    pub fn parse<'context>(
        &'context self,
        expression: &str,
    ) -> Result<FmpzMPoly<'context>, FlintError> {
        FmpzMPoly::parse(self, expression)
    }

    /// Returns the number of variables in this context.
    pub fn variable_count(&self) -> usize {
        self.variable_pointers.len()
    }

    fn as_ptr(&self) -> *const ffi::fmpz_mpoly_ctx_struct {
        self.raw.as_ref()
    }

    fn variable_pointers(&self) -> Vec<*const c_char> {
        self.variable_pointers.clone()
    }
}

impl Drop for FmpzMPolyContext {
    fn drop(&mut self) {
        // SAFETY: The context was initialized by FLINT. Polynomial lifetimes
        // ensure every value using it has already been dropped.
        unsafe { ffi::fmpz_mpoly_ctx_clear(self.raw.as_mut()) };
    }
}

/// An owned FLINT multivariate polynomial over the integers.
pub struct FmpzMPoly<'context> {
    raw: Box<ffi::fmpz_mpoly_struct>,
    context: &'context FmpzMPolyContext,
}

impl<'context> FmpzMPoly<'context> {
    /// Creates the zero polynomial in `context`.
    pub fn zero(context: &'context FmpzMPolyContext) -> Self {
        let mut raw = Box::<ffi::fmpz_mpoly_struct>::new_uninit();

        // SAFETY: `raw` is writable storage and `context` is a live, initialized
        // FLINT context which outlives the returned polynomial.
        unsafe { ffi::fmpz_mpoly_init(raw.as_mut_ptr(), context.as_ptr()) };

        Self {
            // SAFETY: `fmpz_mpoly_init` completed initialization above.
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    /// Parses a polynomial using the names and monomial order in `context`.
    pub fn parse(
        context: &'context FmpzMPolyContext,
        expression: &str,
    ) -> Result<Self, FlintError> {
        let expression = encode_expression(expression)?;
        let mut result = Self::zero(context);
        let mut variables = context.variable_pointers();

        // SAFETY: All pointers refer to live C strings and initialized FLINT
        // values. `variables` remains alive for the duration of the call.
        let status = unsafe {
            ffi::fmpz_mpoly_set_str_pretty(
                result.raw.as_mut(),
                expression.as_ptr(),
                variables.as_mut_ptr(),
                context.as_ptr(),
            )
        };
        if status != 0 {
            return Err(FlintError::new(format!(
                "FLINT could not parse integer polynomial {expression:?}"
            )));
        }
        Ok(result)
    }

    /// Raises this polynomial to a nonnegative integer power.
    pub fn pow(&self, exponent: u64) -> Result<Self, FlintError> {
        let exponent = ffi::ulong::try_from(exponent)
            .map_err(|_| FlintError::new("polynomial exponent does not fit a FLINT ulong"))?;
        let mut result = Self::zero(self.context);

        // SAFETY: The input, output, and context are initialized and belong to
        // the same context. The output is distinct from the input.
        let status = unsafe {
            ffi::fmpz_mpoly_pow_ui(
                result.raw.as_mut(),
                self.raw.as_ref(),
                exponent,
                self.context.as_ptr(),
            )
        };
        if status == 0 {
            return Err(FlintError::new("FLINT integer polynomial power failed"));
        }
        Ok(result)
    }

    /// Adds a signed machine-size integer constant and returns a new polynomial.
    pub fn add_si(&self, constant: ffi::slong) -> Self {
        let mut result = Self::zero(self.context);

        // SAFETY: The input, output, and context are initialized and belong to
        // the same context. The output is distinct from the input.
        unsafe {
            ffi::fmpz_mpoly_add_si(
                result.raw.as_mut(),
                self.raw.as_ref(),
                constant,
                self.context.as_ptr(),
            );
        }
        result
    }

    /// Multiplies two polynomials and returns a newly initialized product.
    pub fn mul(&self, right: &Self) -> Self {
        self.assert_same_context(right);
        let mut result = Self::zero(self.context);

        // SAFETY: Both inputs and the fresh output use the same initialized
        // context, and the output aliases neither input.
        unsafe {
            ffi::fmpz_mpoly_mul(
                result.raw.as_mut(),
                self.raw.as_ref(),
                right.raw.as_ref(),
                self.context.as_ptr(),
            );
        }
        result
    }

    /// Divides by `divisor` when the quotient is exact.
    pub fn exact_div(&self, divisor: &Self) -> Result<Self, FlintError> {
        self.assert_same_context(divisor);
        let mut result = Self::zero(self.context);

        // SAFETY: The dividend, divisor, and fresh quotient use the same live
        // context, and the quotient aliases neither input.
        let status = unsafe {
            ffi::fmpz_mpoly_divides(
                result.raw.as_mut(),
                self.raw.as_ref(),
                divisor.raw.as_ref(),
                self.context.as_ptr(),
            )
        };
        if status == 0 {
            return Err(FlintError::new(
                "FLINT integer polynomial division was not exact",
            ));
        }
        Ok(result)
    }

    /// Computes the resultant with respect to the zero-based variable index.
    pub fn resultant(&self, right: &Self, variable: usize) -> Result<Self, FlintError> {
        self.assert_same_context(right);
        if variable >= self.context.variable_count() {
            return Err(FlintError::new("resultant variable index is out of range"));
        }
        let mut result = Self::zero(self.context);

        // SAFETY: Both inputs and the fresh output use the same initialized
        // context, and `variable` names a variable in that context.
        let status = unsafe {
            ffi::fmpz_mpoly_resultant(
                result.raw.as_mut(),
                self.raw.as_ref(),
                right.raw.as_ref(),
                variable as ffi::slong,
                self.context.as_ptr(),
            )
        };
        if status == 0 {
            return Err(FlintError::new("FLINT integer polynomial resultant failed"));
        }
        Ok(result)
    }

    /// Computes a GCD with the selected FLINT implementation.
    pub fn gcd(&self, right: &Self, algorithm: GcdAlgorithm) -> Result<Self, FlintError> {
        self.assert_same_context(right);
        let mut result = Self::zero(self.context);

        // SAFETY: Both inputs and the fresh output use the same initialized
        // context, and the output aliases neither input.
        let status = unsafe {
            match algorithm {
                GcdAlgorithm::Auto => ffi::fmpz_mpoly_gcd(
                    result.raw.as_mut(),
                    self.raw.as_ref(),
                    right.raw.as_ref(),
                    self.context.as_ptr(),
                ),
                GcdAlgorithm::Hensel => ffi::fmpz_mpoly_gcd_hensel(
                    result.raw.as_mut(),
                    self.raw.as_ref(),
                    right.raw.as_ref(),
                    self.context.as_ptr(),
                ),
                GcdAlgorithm::Zippel => ffi::fmpz_mpoly_gcd_zippel(
                    result.raw.as_mut(),
                    self.raw.as_ref(),
                    right.raw.as_ref(),
                    self.context.as_ptr(),
                ),
                GcdAlgorithm::Zippel2 => ffi::fmpz_mpoly_gcd_zippel2(
                    result.raw.as_mut(),
                    self.raw.as_ref(),
                    right.raw.as_ref(),
                    self.context.as_ptr(),
                ),
            }
        };
        if status == 0 {
            return Err(FlintError::new(format!(
                "FLINT integer polynomial {algorithm:?} GCD failed"
            )));
        }
        Ok(result)
    }

    /// Computes the complete irreducible factorization over the integers.
    pub fn factor(&self) -> Result<FmpzMPolyFactor<'context>, FlintError> {
        let mut result = FmpzMPolyFactor::new(self.context);

        // SAFETY: The input, factorization output, and context are initialized,
        // and the output is distinct from the input polynomial.
        let status = unsafe {
            ffi::fmpz_mpoly_factor(
                result.raw.as_mut(),
                self.raw.as_ref(),
                self.context.as_ptr(),
            )
        };
        if status == 0 {
            return Err(FlintError::new(
                "FLINT integer polynomial factorization failed",
            ));
        }
        Ok(result)
    }

    /// Returns the number of nonzero terms.
    pub fn len(&self) -> usize {
        // SAFETY: The polynomial and its context are initialized and live.
        unsafe { ffi::fmpz_mpoly_length(self.raw.as_ref(), self.context.as_ptr()) as usize }
    }

    /// Tests equality between polynomials in the same context.
    pub fn equals(&self, right: &Self) -> bool {
        self.assert_same_context(right);
        // SAFETY: Both polynomials and their common context are initialized.
        unsafe {
            ffi::fmpz_mpoly_equal(self.raw.as_ref(), right.raw.as_ref(), self.context.as_ptr()) != 0
        }
    }

    /// Formats the polynomial with the context's variable names.
    pub fn to_pretty_string(&self) -> Result<String, FlintError> {
        let mut variables = self.context.variable_pointers();

        // SAFETY: The polynomial, context, and variable strings are live. FLINT
        // returns either null or a null-terminated allocation owned by FLINT.
        let pointer = unsafe {
            ffi::fmpz_mpoly_get_str_pretty(
                self.raw.as_ref(),
                variables.as_mut_ptr(),
                self.context.as_ptr(),
            )
        };
        if pointer.is_null() {
            return Err(FlintError::new(
                "FLINT could not format an integer polynomial",
            ));
        }

        // SAFETY: `pointer` is non-null and points to the null-terminated string
        // returned above. Copy the bytes before releasing the FLINT allocation.
        let formatted = unsafe { CStr::from_ptr(pointer).to_string_lossy().into_owned() };
        // SAFETY: This pointer was allocated by FLINT's string formatter and has
        // not previously been freed.
        unsafe { ffi::flint_free(pointer.cast()) };
        Ok(formatted)
    }

    fn assert_same_context(&self, right: &Self) {
        assert!(
            std::ptr::eq(self.context, right.context),
            "FLINT polynomial operands belong to different contexts"
        );
    }
}

impl Drop for FmpzMPoly<'_> {
    fn drop(&mut self) {
        // SAFETY: The polynomial was initialized by FLINT and its borrowed
        // context remains live for the duration of this destructor.
        unsafe { ffi::fmpz_mpoly_clear(self.raw.as_mut(), self.context.as_ptr()) };
    }
}

impl fmt::Debug for FmpzMPoly<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_pretty_string() {
            Ok(polynomial) => formatter.write_str(&polynomial),
            Err(_) => formatter
                .debug_struct("FmpzMPoly")
                .field("terms", &self.len())
                .finish(),
        }
    }
}

/// An owned FLINT factorization of a multivariate integer polynomial.
pub struct FmpzMPolyFactor<'context> {
    raw: Box<ffi::fmpz_mpoly_factor_struct>,
    context: &'context FmpzMPolyContext,
}

impl<'context> FmpzMPolyFactor<'context> {
    fn new(context: &'context FmpzMPolyContext) -> Self {
        let mut raw = Box::<ffi::fmpz_mpoly_factor_struct>::new_uninit();

        // SAFETY: `raw` points to suitably aligned writable storage, and the
        // context remains live for the lifetime of the returned factorization.
        unsafe { ffi::fmpz_mpoly_factor_init(raw.as_mut_ptr(), context.as_ptr()) };

        Self {
            // SAFETY: `fmpz_mpoly_factor_init` initialized the value above.
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    /// Returns the number of nonconstant irreducible factors.
    pub fn len(&self) -> usize {
        // SAFETY: The factorization and its context are initialized and live.
        unsafe { ffi::fmpz_mpoly_factor_length(self.raw.as_ref(), self.context.as_ptr()) as usize }
    }

    /// Expands the constant, factors, and multiplicities into one polynomial.
    pub fn expand(&self) -> Result<FmpzMPoly<'context>, FlintError> {
        let mut result = FmpzMPoly::zero(self.context);

        // SAFETY: The factorization, fresh output polynomial, and their common
        // context are initialized and live for this call.
        let status = unsafe {
            ffi::fmpz_mpoly_factor_expand(
                result.raw.as_mut(),
                self.raw.as_ref(),
                self.context.as_ptr(),
            )
        };
        if status == 0 {
            return Err(FlintError::new(
                "FLINT could not expand an integer polynomial factorization",
            ));
        }
        Ok(result)
    }
}

impl Drop for FmpzMPolyFactor<'_> {
    fn drop(&mut self) {
        // SAFETY: FLINT initialized the factorization and its borrowed context
        // remains live for the duration of this destructor.
        unsafe { ffi::fmpz_mpoly_factor_clear(self.raw.as_mut(), self.context.as_ptr()) };
    }
}

impl fmt::Debug for FmpzMPolyFactor<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FmpzMPolyFactor")
            .field("factors", &self.len())
            .finish()
    }
}

/// A lexicographically ordered FLINT multivariate polynomial context modulo a word-size modulus.
pub struct NmodMPolyContext {
    raw: Box<ffi::nmod_mpoly_ctx_struct>,
    modulus: ffi::ulong,
    _variable_names: Vec<CString>,
    variable_pointers: Vec<*const c_char>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl NmodMPolyContext {
    /// Creates a modular polynomial context with the supplied variable order.
    ///
    /// GCD and resultant benchmarks must supply a prime modulus.
    pub fn new<S: AsRef<str>>(variable_names: &[S], modulus: u64) -> Result<Self, FlintError> {
        initialize_single_thread();
        let modulus = ffi::ulong::try_from(modulus)
            .map_err(|_| FlintError::new("modulus does not fit a FLINT ulong"))?;
        if modulus < 2 {
            return Err(FlintError::new("FLINT modulus must be at least two"));
        }
        let (variable_names, variable_pointers) = encode_variable_names(variable_names)?;
        let mut raw = Box::<ffi::nmod_mpoly_ctx_struct>::new_uninit();

        // SAFETY: `raw` points to suitably aligned, writable storage, the variable
        // count and modulus fit their FLINT types, and FLINT initializes the context.
        unsafe {
            ffi::nmod_mpoly_ctx_init(
                raw.as_mut_ptr(),
                variable_names.len() as ffi::slong,
                ffi::ordering_t_ORD_LEX,
                modulus,
            );
        }

        Ok(Self {
            // SAFETY: `nmod_mpoly_ctx_init` completed initialization above.
            raw: unsafe { raw.assume_init() },
            modulus,
            _variable_names: variable_names,
            variable_pointers,
            _not_send_or_sync: PhantomData,
        })
    }

    /// Parses a modular polynomial using this context's variable names.
    pub fn parse<'context>(
        &'context self,
        expression: &str,
    ) -> Result<NmodMPoly<'context>, FlintError> {
        NmodMPoly::parse(self, expression)
    }

    /// Creates the zero polynomial in this context.
    pub fn zero(&self) -> NmodMPoly<'_> {
        NmodMPoly::zero(self)
    }

    /// Builds a dense univariate polynomial from coefficients in ascending degree order.
    pub fn dense_univariate(&self, coefficients: &[u64]) -> Result<NmodMPoly<'_>, FlintError> {
        if self.variable_count() != 1 {
            return Err(FlintError::new(
                "dense_univariate requires a one-variable FLINT context",
            ));
        }

        let mut result = self.zero();
        for (degree, &coefficient) in coefficients.iter().enumerate() {
            let coefficient = ffi::ulong::try_from(coefficient % self.modulus())
                .map_err(|_| FlintError::new("coefficient does not fit a FLINT ulong"))?;
            if coefficient == 0 {
                continue;
            }
            let degree = ffi::ulong::try_from(degree)
                .map_err(|_| FlintError::new("degree does not fit a FLINT ulong"))?;
            let exponents = [degree];

            // SAFETY: `result` and its context are initialized, `exponents`
            // supplies the one exponent required by this univariate context, and
            // the coefficient is reduced modulo the context modulus.
            unsafe {
                ffi::nmod_mpoly_push_term_ui_ui(
                    result.raw.as_mut(),
                    coefficient,
                    exponents.as_ptr(),
                    self.as_ptr(),
                )
            };
        }

        // SAFETY: All pushed terms belong to `result` and use this live context.
        // Sorting restores the canonical monomial order expected by operations.
        unsafe { ffi::nmod_mpoly_sort_terms(result.raw.as_mut(), self.as_ptr()) };
        Ok(result)
    }

    /// Returns the context modulus.
    pub fn modulus(&self) -> u64 {
        self.modulus as u64
    }

    /// Returns the number of variables in this context.
    pub fn variable_count(&self) -> usize {
        self.variable_pointers.len()
    }

    fn as_ptr(&self) -> *const ffi::nmod_mpoly_ctx_struct {
        self.raw.as_ref()
    }

    fn variable_pointers(&self) -> Vec<*const c_char> {
        self.variable_pointers.clone()
    }
}

impl Drop for NmodMPolyContext {
    fn drop(&mut self) {
        // SAFETY: The context was initialized by FLINT. Polynomial lifetimes
        // ensure every value using it has already been dropped.
        unsafe { ffi::nmod_mpoly_ctx_clear(self.raw.as_mut()) };
    }
}

/// An owned FLINT multivariate polynomial modulo a word-size modulus.
pub struct NmodMPoly<'context> {
    raw: Box<ffi::nmod_mpoly_struct>,
    context: &'context NmodMPolyContext,
}

impl<'context> NmodMPoly<'context> {
    /// Creates the zero polynomial in `context`.
    pub fn zero(context: &'context NmodMPolyContext) -> Self {
        let mut raw = Box::<ffi::nmod_mpoly_struct>::new_uninit();

        // SAFETY: `raw` is writable storage and `context` is a live, initialized
        // FLINT context which outlives the returned polynomial.
        unsafe { ffi::nmod_mpoly_init(raw.as_mut_ptr(), context.as_ptr()) };

        Self {
            // SAFETY: `nmod_mpoly_init` completed initialization above.
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    /// Parses a polynomial using the names and monomial order in `context`.
    pub fn parse(
        context: &'context NmodMPolyContext,
        expression: &str,
    ) -> Result<Self, FlintError> {
        let expression = encode_expression(expression)?;
        let mut result = Self::zero(context);
        let mut variables = context.variable_pointers();

        // SAFETY: The expression and variable pointers refer to live C strings,
        // and the polynomial and context are initialized for this call.
        let status = unsafe {
            ffi::nmod_mpoly_set_str_pretty(
                result.raw.as_mut(),
                expression.as_ptr(),
                variables.as_mut_ptr(),
                context.as_ptr(),
            )
        };
        if status != 0 {
            return Err(FlintError::new(format!(
                "FLINT could not parse modular polynomial {expression:?}"
            )));
        }
        Ok(result)
    }

    /// Raises this polynomial to a nonnegative integer power.
    pub fn pow(&self, exponent: u64) -> Result<Self, FlintError> {
        let exponent = ffi::ulong::try_from(exponent)
            .map_err(|_| FlintError::new("polynomial exponent does not fit a FLINT ulong"))?;
        let mut result = Self::zero(self.context);

        // SAFETY: The input, output, and context are initialized and belong to
        // the same context. The output is distinct from the input.
        let status = unsafe {
            ffi::nmod_mpoly_pow_ui(
                result.raw.as_mut(),
                self.raw.as_ref(),
                exponent,
                self.context.as_ptr(),
            )
        };
        if status == 0 {
            return Err(FlintError::new("FLINT modular polynomial power failed"));
        }
        Ok(result)
    }

    /// Adds an unsigned machine-size constant and returns a new polynomial.
    pub fn add_ui(&self, constant: u64) -> Result<Self, FlintError> {
        let constant = ffi::ulong::try_from(constant)
            .map_err(|_| FlintError::new("constant does not fit a FLINT ulong"))?;
        let mut result = Self::zero(self.context);

        // SAFETY: The input, output, and context are initialized and belong to
        // the same context. The output is distinct from the input.
        unsafe {
            ffi::nmod_mpoly_add_ui(
                result.raw.as_mut(),
                self.raw.as_ref(),
                constant,
                self.context.as_ptr(),
            );
        }
        Ok(result)
    }

    /// Subtracts an unsigned machine-size constant and returns a new polynomial.
    pub fn sub_ui(&self, constant: u64) -> Result<Self, FlintError> {
        let constant = ffi::ulong::try_from(constant)
            .map_err(|_| FlintError::new("constant does not fit a FLINT ulong"))?;
        let mut result = Self::zero(self.context);

        // SAFETY: The input, output, and context are initialized and belong to
        // the same context. The output is distinct from the input.
        unsafe {
            ffi::nmod_mpoly_sub_ui(
                result.raw.as_mut(),
                self.raw.as_ref(),
                constant,
                self.context.as_ptr(),
            );
        }
        Ok(result)
    }

    /// Multiplies two polynomials and returns a newly initialized product.
    pub fn mul(&self, right: &Self) -> Self {
        self.assert_same_context(right);
        let mut result = Self::zero(self.context);

        // SAFETY: Both inputs and the fresh output use the same initialized
        // context, and the output aliases neither input.
        unsafe {
            ffi::nmod_mpoly_mul(
                result.raw.as_mut(),
                self.raw.as_ref(),
                right.raw.as_ref(),
                self.context.as_ptr(),
            );
        }
        result
    }

    /// Returns the number of nonzero terms.
    pub fn len(&self) -> usize {
        // SAFETY: The polynomial and its context are initialized and live.
        unsafe { ffi::nmod_mpoly_length(self.raw.as_ref(), self.context.as_ptr()) as usize }
    }

    /// Formats the polynomial with the context's variable names.
    pub fn to_pretty_string(&self) -> Result<String, FlintError> {
        let mut variables = self.context.variable_pointers();

        // SAFETY: The polynomial, context, and variable strings are live. FLINT
        // returns either null or a null-terminated allocation owned by FLINT.
        let pointer = unsafe {
            ffi::nmod_mpoly_get_str_pretty(
                self.raw.as_ref(),
                variables.as_mut_ptr(),
                self.context.as_ptr(),
            )
        };
        if pointer.is_null() {
            return Err(FlintError::new(
                "FLINT could not format a modular polynomial",
            ));
        }

        // SAFETY: `pointer` is non-null and points to the null-terminated string
        // returned above. Copy the bytes before releasing the FLINT allocation.
        let formatted = unsafe { CStr::from_ptr(pointer).to_string_lossy().into_owned() };
        // SAFETY: This pointer was allocated by FLINT's string formatter and has
        // not previously been freed.
        unsafe { ffi::flint_free(pointer.cast()) };
        Ok(formatted)
    }

    fn assert_same_context(&self, right: &Self) {
        assert!(
            std::ptr::eq(self.context, right.context),
            "FLINT polynomial operands belong to different contexts"
        );
    }
}

impl Drop for NmodMPoly<'_> {
    fn drop(&mut self) {
        // SAFETY: The polynomial was initialized by FLINT and its borrowed
        // context remains live for the duration of this destructor.
        unsafe { ffi::nmod_mpoly_clear(self.raw.as_mut(), self.context.as_ptr()) };
    }
}

impl fmt::Debug for NmodMPoly<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_pretty_string() {
            Ok(polynomial) => formatter.write_str(&polynomial),
            Err(_) => formatter
                .debug_struct("NmodMPoly")
                .field("terms", &self.len())
                .field("modulus", &self.context.modulus())
                .finish(),
        }
    }
}
