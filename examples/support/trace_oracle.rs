//! Optional benchmark-only binding to the shared Ratracer interpreter.
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Arc;
use symbolica::domains::finite_field::{FiniteFieldCore, FiniteFieldElement, Zp64};

pub struct TraceOracle {
    handle: *mut c_void,
    evaluate: unsafe extern "C" fn(*mut c_void, u64, *const u64, usize, *mut u64) -> i32,
    close: unsafe extern "C" fn(*mut c_void),
    count: unsafe extern "C" fn(*mut c_void) -> u64,
    input: Vec<u64>,
    _library: Arc<libloading::Library>,
}

impl TraceOracle {
    pub fn from_env(names: &[String]) -> Option<Self> {
        let path = std::env::var("TRACE_ORACLE_PATH").ok()?;
        let library = std::env::var("TRACE_ORACLE_LIBRARY").expect("TRACE_ORACLE_LIBRARY");
        // The environment selects our local benchmark ABI, never a library
        // supplied by an oracle expression. Keep it loaded until after close.
        unsafe {
            let library = Arc::new(libloading::Library::new(library).unwrap());
            let open: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> *mut c_void> =
                library.get(b"rr_open").unwrap();
            let inputs: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> usize> =
                library.get(b"rr_inputs").unwrap();
            let name: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, usize) -> *const c_char,
            > = library.get(b"rr_input_name").unwrap();
            let handle = open(CString::new(path).unwrap().as_ptr());
            assert!(
                !handle.is_null(),
                "cannot load finalized single-output trace"
            );
            let oracle = Self {
                handle,
                evaluate: *library.get(b"rr_evaluate").unwrap(),
                close: *library.get(b"rr_close").unwrap(),
                count: *library.get(b"rr_calls").unwrap(),
                input: vec![0; names.len()],
                _library: library.clone(),
            };
            assert_eq!(inputs(handle), names.len());
            for (i, expected) in names.iter().enumerate() {
                let actual = name(handle, i);
                assert!(!actual.is_null());
                assert_eq!(CStr::from_ptr(actual).to_str().unwrap(), expected);
            }
            Some(oracle)
        }
    }
    pub fn evaluate(
        &mut self,
        field: &Zp64,
        point: &[FiniteFieldElement<u64>],
    ) -> Option<FiniteFieldElement<u64>> {
        assert_eq!(point.len(), self.input.len());
        for (value, x) in self.input.iter_mut().zip(point) {
            *value = field.from_element(x);
        }
        let mut output = 0;
        let status = unsafe {
            (self.evaluate)(
                self.handle,
                field.get_prime(),
                self.input.as_ptr(),
                self.input.len(),
                &mut output,
            )
        };
        match status {
            0 => {
                assert!(output < field.get_prime());
                Some(field.to_element(output))
            }
            1 => None,
            _ => panic!("Ratracer interpreter error or unsupported prime"),
        }
    }
    pub fn calls(&self) -> usize {
        unsafe { (self.count)(self.handle) as usize }
    }
}
impl Drop for TraceOracle {
    fn drop(&mut self) {
        unsafe { (self.close)(self.handle) };
    }
}
