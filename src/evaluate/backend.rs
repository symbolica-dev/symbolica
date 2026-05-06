use super::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub struct ExportedCode<T: CompiledNumber> {
    pub(super) path: PathBuf,
    pub(super) function_name: String,
    pub(super) _phantom: std::marker::PhantomData<T>,
}

/// Represents a library that can be loaded with [Self::load].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub struct CompiledCode<T: CompiledNumber> {
    path: PathBuf,
    function_name: String,
    _phantom: std::marker::PhantomData<T>,
}

/// Maximum length stored in the error message buffer
pub(crate) const CUDA_ERRMSG_LEN: usize = 256;
/// Struct representing the data created for the CUDA evaluation.
#[repr(C)]
pub struct CudaEvaluationData {
    pub params: *mut c_void,
    pub out: *mut c_void,
    pub n: usize,             // Number of evaluations
    pub block_size: usize,    // Number of threads per block
    pub in_dimension: usize,  // Number of input parameters
    pub out_dimension: usize, // Number of output parameters
    pub last_error: i32,
    pub errmsg: [std::os::raw::c_char; CUDA_ERRMSG_LEN],
}

impl CudaEvaluationData {
    pub fn check_for_error(&self) -> Result<(), String> {
        unsafe {
            if self.last_error != 0 {
                let err_msg = std::ffi::CStr::from_ptr(self.errmsg.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                return Err(format!("CUDA error: {}", err_msg));
            }
        }
        Ok(())
    }
}

/// Settings for CUDA.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub struct CudaLoadSettings {
    pub number_of_evaluations: usize,
    /// The number of threads per block for CUDA evaluation.
    pub block_size: usize,
}

impl Default for CudaLoadSettings {
    fn default() -> Self {
        CudaLoadSettings {
            number_of_evaluations: 1,
            block_size: 256, // default CUDA block size
        }
    }
}

impl<T: CompiledNumber> CompiledCode<T> {
    /// Load the evaluator from the compiled shared library.
    pub fn load(&self) -> Result<T::Evaluator, String> {
        T::Evaluator::load(&self.path, &self.function_name)
    }

    /// Load the evaluator from the compiled shared library.
    pub fn load_with_settings(&self, settings: T::Settings) -> Result<T::Evaluator, String> {
        T::Evaluator::load_with_settings(&self.path, &self.function_name, settings)
    }
}

type EvalTypeWithBuffer<'a, T> =
    libloading::Symbol<'a, unsafe extern "C" fn(params: *const T, buffer: *mut T, out: *mut T)>;
type CudaEvalType<'a, T> = libloading::Symbol<
    'a,
    unsafe extern "C" fn(params: *const T, out: *mut T, data: *const CudaEvaluationData),
>;
type CudaInitDataType<'a> = libloading::Symbol<
    'a,
    unsafe extern "C" fn(n: usize, block_size: usize) -> *const CudaEvaluationData,
>;
type CudaDestroyDataType<'a> =
    libloading::Symbol<'a, unsafe extern "C" fn(data: *const CudaEvaluationData) -> i32>;
type GetBufferLenType<'a> = libloading::Symbol<'a, unsafe extern "C" fn() -> c_ulong>;

struct EvaluatorFunctionsRealf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, f64>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsRealf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = f64::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, f64> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsRealf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

type L = std::sync::Arc<libloading::Library>;

self_cell!(
    struct LibraryRealf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsRealf64,
    }
);

struct EvaluatorFunctionsSimdRealf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, wide::f64x4>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsSimdRealf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = wide::f64x4::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, wide::f64x4> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsSimdRealf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

self_cell!(
    struct LibrarySimdComplexf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsSimdComplexf64,
    }
);

struct EvaluatorFunctionsSimdComplexf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, Complex<wide::f64x4>>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsSimdComplexf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = Complex::<wide::f64x4>::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, Complex<wide::f64x4>> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsSimdComplexf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

self_cell!(
    struct LibrarySimdRealf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsSimdRealf64,
    }
);

struct EvaluatorFunctionsComplexf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, Complex<f64>>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsComplexf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = Complex::<f64>::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, Complex<f64>> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsComplexf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

self_cell!(
    struct LibraryComplexf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsComplexf64,
    }
);

struct EvaluatorFunctionsCudaRealf64<'lib> {
    eval: CudaEvalType<'lib, f64>,
    init_data: CudaInitDataType<'lib>,
    destroy_data: CudaDestroyDataType<'lib>,
}

impl<'lib> EvaluatorFunctionsCudaRealf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = CudaRealf64::construct_function_name(function_name);
        unsafe {
            let eval: CudaEvalType<'lib, f64> = lib
                .get(format!("{}_vec", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let init_data: CudaInitDataType<'lib> = lib
                .get(format!("{}_init_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let destroy_data: CudaDestroyDataType<'lib> = lib
                .get(format!("{}_destroy_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsCudaRealf64 {
                eval,
                init_data,
                destroy_data,
            })
        }
    }
}

self_cell!(
    struct LibraryCudaRealf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsCudaRealf64,
    }
);

struct EvaluatorFunctionsCudaComplexf64<'lib> {
    eval: CudaEvalType<'lib, Complex<f64>>,
    init_data: CudaInitDataType<'lib>,
    destroy_data: CudaDestroyDataType<'lib>,
}

impl<'lib> EvaluatorFunctionsCudaComplexf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = CudaComplexf64::construct_function_name(function_name);
        unsafe {
            let eval: CudaEvalType<'lib, Complex<f64>> = lib
                .get(format!("{}_vec", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let init_data: CudaInitDataType<'lib> = lib
                .get(format!("{}_init_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let destroy_data: CudaDestroyDataType<'lib> = lib
                .get(format!("{}_destroy_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsCudaComplexf64 {
                eval,
                init_data,
                destroy_data,
            })
        }
    }
}

self_cell!(
    struct LibraryCudaComplexf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsCudaComplexf64,
    }
);

impl ExpressionEvaluator<Complex<Rational>> {
    /// JIT-compiles the evaluator using SymJIT.
    ///
    /// You can supply the types [f64], [wide::f64x4] for SIMD, [Complex] over [f64] and [wide::f64x4] for Complex SIMD.
    ///
    /// # Examples
    ///
    /// Compile and evaluate the function `x + y` for `f64` inputs:
    /// ```rust
    /// # use symbolica::{atom::AtomCore, parse};
    /// # use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let params = vec![parse!("x"), parse!("y")];
    /// let mut evaluator = parse!("x + y")
    ///     .evaluator(&FunctionMap::new(), &params, OptimizationSettings::default())
    ///     .unwrap()
    ///     .jit_compile::<f64>()
    ///     .unwrap();
    ///
    /// let mut res = [0.];
    /// evaluator.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    pub fn jit_compile<T: JITCompiledNumber + EvaluationDomain>(
        &self,
    ) -> Result<JITCompiledEvaluator<T>, String> {
        let (instructions, _, constants) = self.export_instructions();
        let constants = constants
            .into_iter()
            .map(|c| symjit::Complex::new(c.re.to_f64(), c.im.to_f64()))
            .collect::<Vec<_>>();

        let external_fns = self
            .external_fns
            .iter()
            .map(|f| {
                let mapped = f.map::<T>();
                if mapped.constant_index.is_none() && mapped.imp.is_none() {
                    return Err(format!(
                        "External function '{}' does not have an implementation",
                        f
                    ));
                }

                Ok(mapped)
            })
            .collect::<Result<Vec<_>, _>>()?;
        T::jit_compile(instructions, constants, &external_fns)
    }
}

impl<T: JITCompiledNumber + Clone> ExpressionEvaluator<T> {
    /// JIT-compiles the evaluator using SymJIT.
    ///
    /// # Examples
    ///
    /// Compile and evaluate the function `x + y` for `f64` inputs:
    /// ```rust
    /// # use symbolica::{atom::AtomCore, parse};
    /// # use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let params = vec![parse!("x"), parse!("y")];
    /// let mut evaluator = parse!("x + y")
    ///     .evaluator(&FunctionMap::new(), &params, OptimizationSettings::default())
    ///     .unwrap()
    ///     .jit_compile::<f64>()
    ///     .unwrap();
    ///
    /// let mut res = [0.];
    /// evaluator.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    pub fn jit_compile(&self) -> Result<JITCompiledEvaluator<T>, String> {
        let (instructions, _, constants) = self.export_instructions();
        let constants = constants
            .into_iter()
            .map(|c| c.to_complex_f64())
            .collect::<Result<Vec<_>, _>>()?;

        let external_fns = self
            .external_fns
            .iter()
            .map(|f| {
                if f.constant_index.is_none() && f.imp.is_none() {
                    return Err(format!(
                        "External function '{}' does not have an implementation",
                        f
                    ));
                }

                Ok(f.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        T::jit_compile(instructions, constants, &external_fns)
    }
}

fn translate_to_symjit(
    instructions: Vec<Instruction>,
    constants: Vec<symjit::Complex<f64>>,
    config: Config,
) -> Result<Translator, String> {
    let mut translator = Translator::new(config);

    for z in constants {
        translator.append_constant(z).unwrap();
    }

    fn slot(s: Slot) -> symjit::Slot {
        match s {
            Slot::Param(id) => symjit::Slot::Param(id),
            Slot::Out(id) => symjit::Slot::Out(id),
            Slot::Const(id) => symjit::Slot::Const(id),
            Slot::Temp(id) => symjit::Slot::Temp(id),
        }
    }

    fn slot_list(v: &[Slot]) -> Vec<symjit::Slot> {
        v.iter().map(|s| slot(*s)).collect::<Vec<symjit::Slot>>()
    }

    for q in instructions {
        match q {
            Instruction::Add(lhs, args, num_reals) => translator
                .append_add(&slot(lhs), &slot_list(&args), num_reals)
                .unwrap(),
            Instruction::Mul(lhs, args, num_reals) => translator
                .append_mul(&slot(lhs), &slot_list(&args), num_reals)
                .unwrap(),
            Instruction::Pow(lhs, arg, p, is_real) => translator
                .append_pow(&slot(lhs), &slot(arg), p, is_real)
                .unwrap(),
            Instruction::Powf(lhs, arg, p, is_real) => translator
                .append_powf(&slot(lhs), &slot(arg), &slot(p), is_real)
                .unwrap(),
            Instruction::Assign(lhs, rhs) => {
                translator.append_assign(&slot(lhs), &slot(rhs)).unwrap()
            }
            Instruction::Fun(lhs, fun, is_real) => {
                let (name, tags, args) = *fun;

                let mut name = name.get_ascii_name().ok_or_else(|| {
                    format!(
                        "No ASCII name for symbol {name} available, which is needed for exporting"
                    )
                })?;

                for t in tags {
                    name += &format!("_{}", t);
                }

                translator
                    .append_fun(&slot(lhs), &name, &slot_list(&args), is_real)
                    .map_err(|e| e.to_string())?;
            }
            Instruction::Join(lhs, cond, true_val, false_val) => translator
                .append_join(&slot(lhs), &slot(cond), &slot(true_val), &slot(false_val))
                .unwrap(),
            Instruction::Label(id) => translator.append_label(id).unwrap(),
            Instruction::IfElse(cond, id) => translator.append_if_else(&slot(cond), id).unwrap(),
            Instruction::Goto(id) => translator.append_goto(id).unwrap(),
        }
    }

    Ok(translator)
}

pub trait JITCompiledNumber: Sized {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String>;

    fn convert_external_functions(
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<symjit::Defuns, String>;

    /// Create a JIT-compiled evaluator for this number type.
    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<JITCompiledEvaluator<Self>, String>;

    fn evaluate(eval: &mut JITCompiledEvaluator<Self>, args: &[Self], out: &mut [Self]);
}

impl JITCompiledNumber for f64 {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        Ok(symjit::Complex::new(*self, 0.))
    }

    fn convert_external_functions(
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<symjit::Defuns, String> {
        let mut defuns = Defuns::new();

        for f in external_functions {
            if f.constant_index.is_some() {
                continue;
            }

            let Some(imp) = f.imp.clone() else {
                return Err(format!(
                    "External function '{}' does not have an implementation",
                    f
                ));
            };

            let r: Box<Box<dyn Fn(&[Self]) -> Self + Send + Sync>> = Box::new(imp);

            defuns
                .add_sliced_func(f.export_name(), r)
                .map_err(|e| e.to_string())?;
        }

        Ok(defuns)
    }

    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: &[ExternalFunctionContainer<f64>],
    ) -> Result<JITCompiledEvaluator<Self>, String> {
        if constants.iter().any(|x| x.im != 0.) {
            return Err("complex constants are not supported for f64 JIT export".to_string());
        }

        let mut config = Config::default();
        config.set_complex(false);
        config.set_simd(false);
        config.set_defuns(Self::convert_external_functions(external_functions)?);

        let mut translator = translate_to_symjit(instructions, constants, config)?;

        let app = translator.compile().map_err(|e| e.to_string())?;
        let mut compressed_ir = Vec::new();
        app.save(&mut compressed_ir).map_err(|e| e.to_string())?;

        Ok(JITCompiledEvaluator {
            code: app.seal().map_err(|e| e.to_string())?,
            external_functions: external_functions.to_vec(),
            compressed_ir,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    #[inline(always)]
    fn evaluate(eval: &mut JITCompiledEvaluator<Self>, args: &[Self], out: &mut [Self]) {
        eval.code.evaluate(args, out);
    }
}

/// A JIT-compiled evaluator for expressions, using the SymJIT compiler.
#[derive(Clone)]
pub struct JITCompiledEvaluator<T> {
    code: Applet,
    #[allow(dead_code)]
    external_functions: Vec<ExternalFunctionContainer<T>>,
    #[allow(dead_code)]
    compressed_ir: Vec<u8>,
    batch_input_buffer: Vec<T>,
    batch_output_buffer: Vec<T>,
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for JITCompiledEvaluator<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.external_functions, &self.compressed_ir).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: JITCompiledNumber + EvaluationDomain + symjit::Element + Copy> serde::Deserialize<'de>
    for JITCompiledEvaluator<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (fs, compressed_ir): (Vec<ExternalFunctionContainer<T>>, Vec<u8>) =
            serde::Deserialize::deserialize(deserializer)?;
        Self::load(compressed_ir, fs).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl<T> bincode::Encode for JITCompiledEvaluator<T> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.external_functions, encoder)?;
        bincode::Encode::encode(&self.compressed_ir, encoder)?;
        Ok(())
    }
}

#[cfg(feature = "bincode")]
impl<Context, T: JITCompiledNumber + EvaluationDomain> bincode::Decode<Context>
    for JITCompiledEvaluator<T>
{
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let fs: Vec<ExternalFunctionContainer<T>> = bincode::Decode::decode(decoder)?;
        let compressed_ir: Vec<u8> = bincode::Decode::decode(decoder)?;
        Self::load(compressed_ir, fs).map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

#[cfg(feature = "bincode")]
impl<'de, Context, T: JITCompiledNumber + EvaluationDomain> bincode::BorrowDecode<'de, Context>
    for JITCompiledEvaluator<T>
{
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        <Self as bincode::Decode<Context>>::decode(decoder)
    }
}

impl<T: JITCompiledNumber> JITCompiledEvaluator<T> {
    /// Evaluate the JIT compiled code.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[T], out: &mut [T]) {
        T::evaluate(self, args, out);
    }
}

impl<T: JITCompiledNumber> JITCompiledEvaluator<T> {
    #[allow(dead_code)]
    fn load(
        compressed_ir: Vec<u8>,
        external_functions: Vec<ExternalFunctionContainer<T>>,
    ) -> Result<Self, String> {
        let mut config = Config::default();
        config.set_defuns(T::convert_external_functions(&external_functions)?);

        let app = symjit::Application::load(&mut compressed_ir.as_slice(), &config)
            .map_err(|e| e.to_string())?
            .seal()
            .map_err(|e| e.to_string())?;
        Ok(JITCompiledEvaluator {
            code: app,
            external_functions,
            compressed_ir,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }
}

impl BatchEvaluator<f64> for JITCompiledEvaluator<f64> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

impl JITCompiledNumber for wide::f64x4 {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        let a = self.as_array();
        if !a.iter().all(|x| *x == a[0]) {
            return Err(format!("SIMD value {:?} is not a scalar", self));
        }

        Ok(symjit::Complex::new(a[0], 0.))
    }

    fn convert_external_functions(
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<symjit::Defuns, String> {
        let mut defuns = Defuns::new();
        for f in external_functions {
            if f.constant_index.is_some()
                || f.symbol.is_builtin() && f.symbol.get_evaluation_info().is_none()
            {
                continue;
            }

            let Some(imp) = f.imp.clone() else {
                return Err(format!(
                    "External function '{}' does not have an implementation",
                    f
                ));
            };

            let r: Box<Box<dyn Fn(&[Self]) -> Self + Send + Sync>> = Box::new(imp);

            defuns
                .add_sliced_func(f.export_name(), r)
                .map_err(|e| e.to_string())?;
        }
        Ok(defuns)
    }

    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<JITCompiledEvaluator<Self>, String> {
        let mut config = Config::default();
        config.set_complex(false);
        config.set_simd(true);
        config.set_defuns(Self::convert_external_functions(external_functions)?);

        let mut translator = translate_to_symjit(instructions, constants, config)?;

        let app = translator.compile().map_err(|e| e.to_string())?;
        let mut compressed_ir = Vec::new();
        app.save(&mut compressed_ir).map_err(|e| e.to_string())?;

        Ok(JITCompiledEvaluator {
            code: app.seal().map_err(|e| e.to_string())?,
            external_functions: external_functions.to_vec(),
            compressed_ir,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    #[inline(always)]
    fn evaluate(
        eval: &mut JITCompiledEvaluator<wide::f64x4>,
        args: &[wide::f64x4],
        out: &mut [wide::f64x4],
    ) {
        eval.code.evaluate(args, out);
    }
}

impl BatchEvaluator<f64> for JITCompiledEvaluator<wide::f64x4> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer
            .resize(batch_size.div_ceil(4) * n_params, wide::f64x4::ZERO);

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        i[j + n_params],
                        i[j + 2 * n_params],
                        i[j + 3 * n_params],
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        if j + n_params < i.len() {
                            i[j + n_params]
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params]
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params]
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer
            .resize(batch_size.div_ceil(4) * n_out, wide::f64x4::ZERO);

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            o.copy_from_slice(&i.as_array()[..o.len()]);
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

impl JITCompiledNumber for Complex<f64> {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        Ok(symjit::Complex::new(self.re, self.im))
    }

    fn convert_external_functions(
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<symjit::Defuns, String> {
        let mut defuns = Defuns::new();

        for f in external_functions {
            if f.constant_index.is_some()
                || f.symbol.is_builtin() && f.symbol.get_evaluation_info().is_none()
            {
                continue;
            }

            let Some(imp) = f.imp.clone() else {
                return Err(format!(
                    "External function '{}' does not have an implementation",
                    f
                ));
            };

            // TODO: implement symjit::Element on numeric::Complex
            let k = Box::new(move |x: &[symjit::Complex<f64>]| {
                let ars = unsafe { std::mem::transmute(x) };
                let res = imp(ars);
                symjit::Complex::new(res.re, res.im)
            });

            defuns
                .add_sliced_func(f.export_name(), k)
                .map_err(|e| e.to_string())?;
        }

        Ok(defuns)
    }

    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<JITCompiledEvaluator<Complex<f64>>, String> {
        let mut config = Config::default();
        config.set_complex(true);
        config.set_simd(false);
        config.set_defuns(Self::convert_external_functions(external_functions)?);

        let mut translator = translate_to_symjit(instructions, constants, config)?;

        let app = translator.compile().map_err(|e| e.to_string())?;
        let mut compressed_ir = Vec::new();
        app.save(&mut compressed_ir).map_err(|e| e.to_string())?;

        Ok(JITCompiledEvaluator {
            code: app.seal().map_err(|e| e.to_string())?,
            external_functions: external_functions.to_vec(),
            compressed_ir,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    fn evaluate(
        eval: &mut JITCompiledEvaluator<Complex<f64>>,
        args: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) {
        let args: &[symjit::Complex<f64>] = unsafe { std::mem::transmute(args) };
        let out: &mut [symjit::Complex<f64>] = unsafe { std::mem::transmute(out) };
        eval.code.evaluate(args, out);
    }
}

impl BatchEvaluator<Complex<f64>> for JITCompiledEvaluator<Complex<f64>> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

impl JITCompiledNumber for Complex<wide::f64x4> {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        let re = self.re.as_array();
        if !re.iter().all(|x| *x == re[0]) {
            return Err(format!("SIMD value {:?} is not a scalar", self));
        }

        let im = self.im.as_array();
        if !im.iter().all(|x| *x == im[0]) {
            return Err(format!("SIMD value {:?} is not a scalar", self));
        }

        Ok(symjit::Complex::new(re[0], im[0]))
    }

    fn convert_external_functions(
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<symjit::Defuns, String> {
        let mut defuns = Defuns::new();

        for f in external_functions {
            if f.constant_index.is_some()
                || f.symbol.is_builtin() && f.symbol.get_evaluation_info().is_none()
            {
                continue;
            }

            let Some(imp) = f.imp.clone() else {
                return Err(format!(
                    "External function '{}' does not have an implementation",
                    f
                ));
            };

            // TODO: implement symjit::Element on numeric::Complex
            let k = Box::new(move |x: &[symjit::Complex<wide::f64x4>]| {
                let ars = unsafe { std::mem::transmute(x) };
                let res = imp(ars);
                symjit::Complex::new(res.re, res.im)
            });

            defuns
                .add_sliced_func(f.export_name(), k)
                .map_err(|e| e.to_string())?;
        }

        Ok(defuns)
    }

    /// JIT-compiles the evaluator using SymJIT.
    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: &[ExternalFunctionContainer<Self>],
    ) -> Result<JITCompiledEvaluator<Self>, String> {
        let mut config = Config::default();
        config.set_complex(true);
        config.set_simd(true);
        config.set_defuns(Self::convert_external_functions(external_functions)?);

        let mut translator = translate_to_symjit(instructions, constants, config)?;

        let app = translator.compile().map_err(|e| e.to_string())?;
        let mut compressed_ir = Vec::new();
        app.save(&mut compressed_ir).map_err(|e| e.to_string())?;

        Ok(JITCompiledEvaluator {
            code: app.seal().map_err(|e| e.to_string())?,
            external_functions: external_functions.to_vec(),
            compressed_ir,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    #[inline(always)]
    fn evaluate(
        eval: &mut JITCompiledEvaluator<Self>,
        args: &[Complex<wide::f64x4>],
        out: &mut [Complex<wide::f64x4>],
    ) {
        let args: &[symjit::Complex<wide::f64x4>] = unsafe { std::mem::transmute(args) };
        let out: &mut [symjit::Complex<wide::f64x4>] = unsafe { std::mem::transmute(out) };

        eval.code.evaluate(args, out);
    }
}

impl BatchEvaluator<Complex<f64>> for JITCompiledEvaluator<Complex<wide::f64x4>> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer.resize(
            batch_size.div_ceil(4) * n_params,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        i[j + n_params].re,
                        i[j + 2 * n_params].re,
                        i[j + 3 * n_params].re,
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        i[j + n_params].im,
                        i[j + 2 * n_params].im,
                        i[j + 3 * n_params].im,
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        if j + n_params < i.len() {
                            i[j + n_params].re
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].re
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].re
                        } else {
                            0.0
                        },
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        if j + n_params < i.len() {
                            i[j + n_params].im
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].im
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].im
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer.resize(
            batch_size.div_ceil(4) * n_out,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            for (j, d) in o.iter_mut().enumerate() {
                d.re = i.re.as_array()[j];
                d.im = i.im.as_array()[j];
            }
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

/// A number type that can be used to call a compiled evaluator.
pub trait CompiledNumber: Sized {
    type Evaluator: EvaluatorLoader<Self>;
    type Settings: Default;
    /// A unique suffix for the evaluation function for this particular number type.
    // NOTE: a rename of any suffix will prevent loading older libraries.
    const SUFFIX: &'static str;

    /// Export an evaluator to C++ code for this number type.
    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String>;

    fn construct_function_name(function_name: &str) -> String {
        format!("{}_{}", function_name, Self::SUFFIX)
    }

    /// Get the default compilation options for C++ code generated
    /// for this number type.
    fn get_default_compile_options() -> CompileOptions;
}

/// Load a compiled evaluator from a shared library, optionally with settings.
pub trait EvaluatorLoader<T: CompiledNumber>: Sized {
    /// Load a compiled evaluator from a shared library.
    fn load(file: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        Self::load_with_settings(file, function_name, T::Settings::default())
    }
    fn load_with_settings(
        file: impl AsRef<Path>,
        function_name: &str,
        settings: T::Settings,
    ) -> Result<Self, String>;
}

/// Batch-evaluate the compiled code with basic types such as [f64] or [`Complex<f64>`],
/// automatically reorganizing the batches if necessary.
pub trait BatchEvaluator<T: CompiledNumber> {
    /// Evaluate the compiled code with batched input with the given input parameters, writing the results to `out`.
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[T],
        out: &mut [T],
    ) -> Result<(), String>;
}

impl CompiledNumber for f64 {
    type Evaluator = CompiledRealEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "realf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(match settings.inline_asm {
            InlineASM::X64 => eval.export_asm_real_str(function_name, &settings),
            InlineASM::AArch64 => eval.export_asm_real_str(function_name, &settings),
            InlineASM::AVX2 => {
                Err("AVX2 not supported for complexf64: use Complex<f64x6> instead".to_owned())?
            }
            InlineASM::None => {
                let r = eval.export_generic_cpp_str(function_name, &settings, NumberClass::RealF64);
                r + format!("\nextern \"C\" {{\n\tvoid {function_name}(double *params, double *buffer, double *out) {{\n\t\t{function_name}_gen(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n").as_str()
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<f64> for CompiledRealEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

impl CompiledNumber for Complex<f64> {
    type Evaluator = CompiledComplexEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "complexf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        Ok(match settings.inline_asm {
            InlineASM::X64 => eval.export_asm_complex_str(function_name, &settings),
            InlineASM::AArch64 => eval.export_asm_complex_str(function_name, &settings),
            InlineASM::AVX2 => {
                Err("AVX2 not supported for complexf64: use Complex<f64x6> instead".to_owned())?
            }
            InlineASM::None => {
                let r =
                    eval.export_generic_cpp_str(function_name, &settings, NumberClass::ComplexF64);
                r + format!("\nextern \"C\" {{\n\tvoid {function_name}(std::complex<double> *params, std::complex<double> *buffer, std::complex<double> *out) {{\n\t\t{function_name}_gen(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n").as_str()
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<Complex<f64>> for CompiledComplexEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

/// Efficient evaluator for compiled real-valued functions.
pub struct CompiledRealEvaluator {
    library: LibraryRealf64,
    path: PathBuf,
    fn_name: String,
    buffer_double: Vec<f64>,
}

impl EvaluatorLoader<f64> for CompiledRealEvaluator {
    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledRealEvaluator::load(path, function_name)
    }
}

impl CompiledRealEvaluator {
    pub fn load_new_function(&self, function_name: &str) -> Result<CompiledRealEvaluator, String> {
        let library = LibraryRealf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsRealf64::new(lib, function_name)
        })?;

        let len = unsafe { (library.borrow_dependent().get_buffer_len)() } as usize;

        Ok(CompiledRealEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer_double: vec![0.; len],
            library,
        })
    }
    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledRealEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibraryRealf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsRealf64::new(lib, function_name)
            })?;

            let len = (library.borrow_dependent().get_buffer_len)() as usize;

            Ok(CompiledRealEvaluator {
                fn_name: function_name.to_string(),
                path: path.as_ref().to_path_buf(),
                buffer_double: vec![0.; len],
                library,
            })
        }
    }
    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[f64], out: &mut [f64]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer_double.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledRealEvaluator {}

impl std::fmt::Debug for CompiledRealEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledRealEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledRealEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledRealEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledRealEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledRealEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledRealEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledRealEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledRealEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledRealEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// Efficient evaluator for compiled complex-valued functions.
pub struct CompiledComplexEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibraryComplexf64,
    buffer_complex: Vec<Complex<f64>>,
}

impl EvaluatorLoader<Complex<f64>> for CompiledComplexEvaluator {
    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledComplexEvaluator::load(path, function_name)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledComplexEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledComplexEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledComplexEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledComplexEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledComplexEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledComplexEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledComplexEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

impl CompiledComplexEvaluator {
    /// Load a new function from the same library.
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledComplexEvaluator, String> {
        let library = LibraryComplexf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsComplexf64::new(lib, function_name)
        })?;

        let len = unsafe { (library.borrow_dependent().get_buffer_len)() } as usize;

        Ok(CompiledComplexEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer_complex: vec![Complex::new_zero(); len],
            library,
        })
    }

    /// Load a compiled evaluator from a shared library.
    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledComplexEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };

            let library = LibraryComplexf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsComplexf64::new(lib, function_name)
            })?;

            let len = (library.borrow_dependent().get_buffer_len)() as usize;

            Ok(CompiledComplexEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                buffer_complex: vec![Complex::default(); len],
                library,
            })
        }
    }
    /// Evaluate the compiled code.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[Complex<f64>], out: &mut [Complex<f64>]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer_complex.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledComplexEvaluator {}

impl std::fmt::Debug for CompiledComplexEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledComplexEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledComplexEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

/// Evaluate 4 double-precision floating point numbers in parallel using SIMD instructions.
/// Make sure you add arguments such as `-march=native` to enable full SIMD support for your platform.
///
/// Failure to add this, may result in only two double-precision numbers being evaluated in parallel.
///
/// The compilation requires the `xsimd` C++ library to be installed.
impl CompiledNumber for wide::f64x4 {
    type Evaluator = CompiledSimdRealEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "simd_realf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(match settings.inline_asm {
            // assume AVX2 for X64
            InlineASM::X64 => eval.export_simd_str(function_name, settings, false, InlineASM::AVX2),
            InlineASM::AArch64 => {
                Err("Inline assembly not supported yet for SIMD f64x4".to_owned())?
            }
            asm @ InlineASM::AVX2 | asm @ InlineASM::None => {
                eval.export_simd_str(function_name, settings, false, asm)
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<f64> for CompiledSimdRealEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer
            .resize(batch_size.div_ceil(4) * n_params, wide::f64x4::ZERO);

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        i[j + n_params],
                        i[j + 2 * n_params],
                        i[j + 3 * n_params],
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        if j + n_params < i.len() {
                            i[j + n_params]
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params]
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params]
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer
            .resize(batch_size.div_ceil(4) * n_out, wide::f64x4::ZERO);

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            o.copy_from_slice(&i.as_array()[..o.len()]);
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

/// Efficient evaluator using simd for compiled real-valued functions.
pub struct CompiledSimdRealEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibrarySimdRealf64,
    buffer: Vec<wide::f64x4>,
    batch_input_buffer: Vec<wide::f64x4>,
    batch_output_buffer: Vec<wide::f64x4>,
}

impl EvaluatorLoader<wide::f64x4> for CompiledSimdRealEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledSimdRealEvaluator::load_with_settings(path, function_name, ())
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledSimdRealEvaluator::load(path, function_name)
    }
}

impl CompiledSimdRealEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledSimdRealEvaluator, String> {
        let library = LibrarySimdRealf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsSimdRealf64::new(lib, function_name)
        })?;

        Ok(CompiledSimdRealEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer: vec![
                wide::f64x4::ZERO;
                unsafe { (library.borrow_dependent().get_buffer_len)() } as usize
            ],
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
            library,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledSimdRealEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibrarySimdRealf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsSimdRealf64::new(lib, function_name)
            })?;

            Ok(CompiledSimdRealEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                buffer: vec![
                    wide::f64x4::ZERO;
                    (library.borrow_dependent().get_buffer_len)() as usize
                ],
                batch_input_buffer: Vec::new(),
                batch_output_buffer: Vec::new(),
                library,
            })
        }
    }

    /// Evaluate the compiled code with 4 double-precision floating point numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[wide::f64x4], out: &mut [wide::f64x4]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledSimdRealEvaluator {}

impl std::fmt::Debug for CompiledSimdRealEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledSimdRealEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledSimdRealEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledSimdRealEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledSimdRealEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledSimdRealEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledSimdRealEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledSimdRealEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledSimdRealEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledSimdRealEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// Evaluate 4 double-precision floating point numbers in parallel using SIMD instructions.
/// Make sure you add arguments such as `-march=native` to enable full SIMD support for your platform.
///
/// Failure to add this, may result in only two double-precision numbers being evaluated in parallel.
///
/// The compilation requires the `xsimd` C++ library to be installed.
impl CompiledNumber for Complex<wide::f64x4> {
    type Evaluator = CompiledSimdComplexEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "simd_complexf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(match settings.inline_asm {
            // assume AVX2 for X64
            InlineASM::X64 => eval.export_simd_str(function_name, settings, true, InlineASM::AVX2),
            InlineASM::AArch64 => {
                Err("X64 inline assembly not supported for SIMD f64x4: use AVX2".to_owned())?
            }
            asm @ InlineASM::AVX2 | asm @ InlineASM::None => {
                eval.export_simd_str(function_name, settings, true, asm)
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<Complex<f64>> for CompiledSimdComplexEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer.resize(
            batch_size.div_ceil(4) * n_params,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        i[j + n_params].re,
                        i[j + 2 * n_params].re,
                        i[j + 3 * n_params].re,
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        i[j + n_params].im,
                        i[j + 2 * n_params].im,
                        i[j + 3 * n_params].im,
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        if j + n_params < i.len() {
                            i[j + n_params].re
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].re
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].re
                        } else {
                            0.0
                        },
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        if j + n_params < i.len() {
                            i[j + n_params].im
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].im
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].im
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer.resize(
            batch_size.div_ceil(4) * n_out,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            for (j, d) in o.iter_mut().enumerate() {
                d.re = i.re.as_array()[j];
                d.im = i.im.as_array()[j];
            }
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

/// Efficient evaluator using simd for compiled complex-valued functions.
pub struct CompiledSimdComplexEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibrarySimdComplexf64,
    buffer: Vec<Complex<wide::f64x4>>,
    batch_input_buffer: Vec<Complex<wide::f64x4>>,
    batch_output_buffer: Vec<Complex<wide::f64x4>>,
}

impl EvaluatorLoader<Complex<wide::f64x4>> for CompiledSimdComplexEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledSimdComplexEvaluator::load_with_settings(path, function_name, ())
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledSimdComplexEvaluator::load(path, function_name)
    }
}

impl CompiledSimdComplexEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledSimdComplexEvaluator, String> {
        let library = LibrarySimdComplexf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsSimdComplexf64::new(lib, function_name)
        })?;

        Ok(CompiledSimdComplexEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer: vec![
                Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO);
                unsafe { (library.borrow_dependent().get_buffer_len)() } as usize
            ],
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
            library,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledSimdComplexEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibrarySimdComplexf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsSimdComplexf64::new(lib, function_name)
            })?;

            Ok(CompiledSimdComplexEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                buffer: vec![
                    Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO);
                    (library.borrow_dependent().get_buffer_len)() as usize
                ],
                batch_input_buffer: Vec::new(),
                batch_output_buffer: Vec::new(),
                library,
            })
        }
    }

    /// Evaluate the compiled code with 4 double-precision floating point numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[Complex<wide::f64x4>], out: &mut [Complex<wide::f64x4>]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledSimdComplexEvaluator {}

impl std::fmt::Debug for CompiledSimdComplexEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledSimdComplexEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledSimdComplexEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledSimdComplexEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledSimdComplexEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledSimdComplexEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledSimdComplexEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledSimdComplexEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledSimdComplexEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledSimdComplexEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// CUDA real number type.
pub struct CudaRealf64 {}

impl CompiledNumber for CudaRealf64 {
    type Evaluator = CompiledCudaRealEvaluator;
    type Settings = CudaLoadSettings;
    const SUFFIX: &'static str = "cuda_realf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(eval.export_cuda_str(function_name, settings, NumberClass::RealF64))
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::cuda()
    }
}

/// CUDA complex number type.
pub struct CudaComplexf64 {}

impl CompiledNumber for CudaComplexf64 {
    type Evaluator = CompiledCudaComplexEvaluator;
    type Settings = CudaLoadSettings;
    const SUFFIX: &'static str = "cuda_complexf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        Ok(eval.export_cuda_str(function_name, settings, NumberClass::ComplexF64))
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::cuda()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledCudaRealEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name, &self.settings).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledCudaRealEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name, settings) =
            <(PathBuf, String, CudaLoadSettings)>::deserialize(deserializer)?;
        CompiledCudaRealEvaluator::load_with_settings(&file, &fn_name, settings)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledCudaRealEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)?;
        bincode::Encode::encode(&self.settings, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledCudaRealEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledCudaRealEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        let settings: CudaLoadSettings = bincode::Decode::decode(decoder)?;
        CompiledCudaRealEvaluator::load(&file, &fn_name, settings)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledCudaComplexEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledCudaComplexEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name, settings) =
            <(PathBuf, String, CudaLoadSettings)>::deserialize(deserializer)?;
        CompiledCudaComplexEvaluator::load(&file, &fn_name, settings)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledCudaComplexEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)?;
        bincode::Encode::encode(&self.settings, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledCudaComplexEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledCudaComplexEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        let settings: CudaLoadSettings = bincode::Decode::decode(decoder)?;
        CompiledCudaComplexEvaluator::load(&file, &fn_name, settings)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// Efficient evaluator using CUDA for compiled real-valued functions.
pub struct CompiledCudaRealEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibraryCudaRealf64,
    settings: CudaLoadSettings,
    data: *const CudaEvaluationData,
}

impl EvaluatorLoader<CudaRealf64> for CompiledCudaRealEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledCudaRealEvaluator::load_with_settings(
            path,
            function_name,
            CudaLoadSettings::default(),
        )
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<Self, String> {
        CompiledCudaRealEvaluator::load(path, function_name, settings)
    }
}

impl BatchEvaluator<f64> for CompiledCudaRealEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if self.settings.number_of_evaluations != batch_size {
            return Err(format!(
                "Number of CUDA evaluations {} does not equal batch size {}",
                self.settings.number_of_evaluations, batch_size
            ));
        }

        self.evaluate(params, out)
    }
}

impl CompiledCudaRealEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledCudaRealEvaluator, String> {
        let library = LibraryCudaRealf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsCudaRealf64::new(lib, function_name)
        })?;
        let data = unsafe {
            let data = (library.borrow_dependent().init_data)(
                self.settings.number_of_evaluations,
                self.settings.block_size,
            );
            (*data).check_for_error()?;
            data
        };

        Ok(CompiledCudaRealEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            library,
            settings: self.settings.clone(),
            data,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<CompiledCudaRealEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibraryCudaRealf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsCudaRealf64::new(lib, function_name)
            })?;

            let data = (library.borrow_dependent().init_data)(
                settings.number_of_evaluations,
                settings.block_size,
            );
            (*data).check_for_error()?;

            Ok(CompiledCudaRealEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                library,
                settings,
                data,
            })
        }
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[f64], out: &mut [f64]) -> Result<(), String> {
        unsafe {
            if args.len() != (*self.data).in_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA args length (={}) does not match the expected input dimension (={}*{}).",
                    args.len(),
                    (*self.data).in_dimension,
                    (*self.data).n
                ));
            }
            if out.len() != (*self.data).out_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA out length (={}) does not match the expected output dimension (={}*{}).",
                    out.len(),
                    (*self.data).out_dimension,
                    (*self.data).n
                ));
            }
            (self.library.borrow_dependent().eval)(args.as_ptr(), out.as_mut_ptr(), self.data);
            (*self.data).check_for_error()?;
        }
        Ok(())
    }
}

/// Efficient evaluator using CUDA for compiled complex-valued functions.
pub struct CompiledCudaComplexEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibraryCudaComplexf64,
    settings: CudaLoadSettings,
    data: *const CudaEvaluationData,
}

impl EvaluatorLoader<CudaComplexf64> for CompiledCudaComplexEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledCudaComplexEvaluator::load_with_settings(
            path,
            function_name,
            CudaLoadSettings::default(),
        )
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<Self, String> {
        CompiledCudaComplexEvaluator::load(path, function_name, settings)
    }
}

impl BatchEvaluator<Complex<f64>> for CompiledCudaComplexEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if self.settings.number_of_evaluations != batch_size {
            return Err(format!(
                "Number of CUDA evaluations {} does not equal batch size {}",
                self.settings.number_of_evaluations, batch_size
            ));
        }

        self.evaluate(params, out)
    }
}

impl CompiledCudaComplexEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledCudaComplexEvaluator, String> {
        let library = LibraryCudaComplexf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsCudaComplexf64::new(lib, function_name)
        })?;

        let data = unsafe {
            let data = (library.borrow_dependent().init_data)(
                self.settings.number_of_evaluations,
                self.settings.block_size,
            );
            (*data).check_for_error()?;
            data
        };
        Ok(CompiledCudaComplexEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            library,
            settings: self.settings.clone(),
            data,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<CompiledCudaComplexEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibraryCudaComplexf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsCudaComplexf64::new(lib, function_name)
            })?;

            let data = (library.borrow_dependent().init_data)(
                settings.number_of_evaluations,
                settings.block_size,
            );
            (*data).check_for_error()?;

            Ok(CompiledCudaComplexEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                library,
                settings,
                data,
            })
        }
    }

    /// Evaluate the compiled code with complex numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(
        &mut self,
        args: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        unsafe {
            if args.len() != (*self.data).in_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA args length (={}) does not match the expected input dimension (={}*{}).",
                    args.len(),
                    (*self.data).in_dimension,
                    (*self.data).n
                ));
            }
            if out.len() != (*self.data).out_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA out length (={}) does not match the expected output dimension (={}*{}).",
                    out.len(),
                    (*self.data).out_dimension,
                    (*self.data).n
                ));
            }
            (self.library.borrow_dependent().eval)(args.as_ptr(), out.as_mut_ptr(), self.data);
            (*self.data).check_for_error()?;
        }
        Ok(())
    }
}

unsafe impl Send for CompiledCudaRealEvaluator {}
unsafe impl Send for CompiledCudaComplexEvaluator {}
unsafe impl Sync for CompiledCudaRealEvaluator {}
unsafe impl Sync for CompiledCudaComplexEvaluator {}

impl std::fmt::Debug for CompiledCudaRealEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledCudaRealEvaluator({})", self.fn_name)
    }
}

impl Drop for CompiledCudaRealEvaluator {
    fn drop(&mut self) {
        unsafe {
            let result = (self.library.borrow_dependent().destroy_data)(self.data);
            if result != 0 {
                error!("Warning: failed to free CUDA memory: {}", result);
            }
        }
    }
}

impl Clone for CompiledCudaRealEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

impl std::fmt::Debug for CompiledCudaComplexEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledCudaComplexEvaluator({})", self.fn_name)
    }
}

impl Drop for CompiledCudaComplexEvaluator {
    fn drop(&mut self) {
        unsafe {
            let result = (self.library.borrow_dependent().destroy_data)(self.data);
            if result != 0 {
                error!("Warning: failed to free CUDA memory: {}", result);
            }
        }
    }
}

impl Clone for CompiledCudaComplexEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

/// Options for compiling exported code.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Clone)]
pub struct CompileOptions {
    pub optimization_level: usize,
    pub fast_math: bool,
    pub unsafe_math: bool,
    /// Compile for the native architecture.
    pub native: bool,
    pub compiler: String,
    /// Arguments for the compiler call. Arguments with spaces
    /// must be split into a separate strings.
    ///
    /// For CUDA, the argument `-x cu` is required.
    pub args: Vec<String>,
}

impl Default for CompileOptions {
    /// Default compile options.
    fn default() -> Self {
        CompileOptions {
            optimization_level: 3,
            fast_math: true,
            unsafe_math: true,
            native: true,
            compiler: "g++".to_string(),
            args: vec![],
        }
    }
}

impl CompileOptions {
    /// Set the compiler to `nvcc`.
    pub fn cuda() -> Self {
        CompileOptions {
            optimization_level: 3,
            fast_math: false,
            unsafe_math: false,
            native: false,
            compiler: "nvcc".to_string(),
            args: vec![],
        }
    }
}

impl ToString for CompileOptions {
    /// Convert the compilation options to the string that would be used
    /// in the compiler call.
    fn to_string(&self) -> String {
        let mut s = self.compiler.clone();

        s += &format!(" -shared -O{}", self.optimization_level);

        let nvcc = self.compiler.contains("nvcc");

        if !nvcc {
            s += " -fPIC";
        } else {
            // order is important here for nvcc
            s += " -Xcompiler -fPIC -x cu";
        }

        if self.fast_math && !nvcc {
            s += " -ffast-math";
        }
        if self.unsafe_math && !nvcc {
            s += " -funsafe-math-optimizations";
        }
        if self.native && !nvcc {
            s += " -march=native";
        }
        for arg in &self.args {
            s += " ";
            s += arg;
        }
        s
    }
}

impl<T: CompiledNumber> ExportedCode<T> {
    /// Create a new exported code object from a source file and function name.
    pub fn new(source_path: impl AsRef<Path>, function_name: String) -> Self {
        ExportedCode {
            path: source_path.as_ref().to_path_buf(),
            function_name,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compile the code to a shared library.
    ///
    /// For CUDA, you may have to specify `-code=sm_XY` for your architecture `XY` in the compiler flags to prevent a potentially long
    /// JIT compilation upon the first evaluation.
    pub fn compile(
        &self,
        out: impl AsRef<Path>,
        options: CompileOptions,
    ) -> Result<CompiledCode<T>, std::io::Error> {
        let mut builder = std::process::Command::new(&options.compiler);
        builder
            .arg("-shared")
            .arg(format!("-O{}", options.optimization_level));

        if !options.compiler.contains("nvcc") {
            builder.arg("-fPIC");
        } else {
            // order is important here for nvcc
            builder.arg("-Xcompiler");
            builder.arg("-fPIC");
            builder.arg("-x");
            builder.arg("cu");
        }
        if options.fast_math && !options.compiler.contains("nvcc") {
            builder.arg("-ffast-math");
        }
        if options.unsafe_math && !options.compiler.contains("nvcc") {
            builder.arg("-funsafe-math-optimizations");
        }

        if options.native && !options.compiler.contains("nvcc") {
            builder.arg("-march=native");
        }

        for c in &options.args {
            builder.arg(c);
        }

        let r = builder
            .arg("-o")
            .arg(out.as_ref())
            .arg(&self.path)
            .output()?;

        if !r.status.success() {
            return Err(std::io::Error::other(format!(
                "Could not compile code: {} {}\n{}",
                builder.get_program().to_string_lossy(),
                builder
                    .get_args()
                    .map(|arg| arg.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                String::from_utf8_lossy(&r.stderr)
            )));
        }

        Ok(CompiledCode {
            path: out.as_ref().to_path_buf(),
            function_name: self.function_name.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

/// The inline assembly mode used to generate fast
/// assembly instructions for mathematical operations.
/// Set to `None` to disable inline assembly.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum InlineASM {
    /// Use instructions suitable for x86_64 machines.
    X64,
    /// Use instructions suitable for x86_64 machines with AVX2 support.
    AVX2,
    /// Use instructions suitable for ARM64 machines.
    AArch64,
    /// Do not generate inline assembly.
    None,
}

impl Default for InlineASM {
    /// Set the assembly mode suitable for the current
    /// architecture.
    fn default() -> Self {
        if cfg!(target_arch = "x86_64") {
            InlineASM::X64
        } else if cfg!(target_arch = "aarch64") {
            InlineASM::AArch64
        } else {
            InlineASM::None
        }
    }
}
