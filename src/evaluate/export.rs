use super::*;

/// A number that can be exported to C++ code.
pub trait ExportNumber {
    /// Export the number as a string.
    fn export(&self) -> String;
    /// Export the number wrapped in a C++ type `T`.
    fn export_wrapped(&self) -> String {
        format!("T({})", self.export())
    }
    /// Export the number wrapped in a C++ type `wrapper`.
    fn export_wrapped_with(&self, wrapper: &str) -> String {
        format!("{wrapper}({})", self.export())
    }
    /// Check if the number is real.
    fn is_real(&self) -> bool;
}

impl ExportNumber for f64 {
    fn export(&self) -> String {
        format!("{:e}", self)
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for F64 {
    fn export(&self) -> String {
        format!("{:e}", self)
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for Float {
    fn export(&self) -> String {
        format!("{:e}", self)
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for Rational {
    fn export(&self) -> String {
        self.to_string()
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl<T: ExportNumber + SingleFloat> ExportNumber for Complex<T> {
    fn export(&self) -> String {
        if self.im.is_zero() {
            self.re.export()
        } else {
            format!("{}, {}", self.re.export(), self.im.export())
        }
    }

    fn is_real(&self) -> bool {
        self.im.is_zero()
    }
}

/// The number class used for exporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberClass {
    RealF64,
    ComplexF64,
}

/// Settings for exporting the evaluation tree to C++ code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportSettings {
    /// Include required `#include` statements in the generated code.
    pub include_header: bool,
    /// Set the inline assembly mode.
    /// With `inline_asm` set to any value other than `None`,
    /// high-performance inline ASM code will be generated for most
    /// evaluation instructions. This often gives better performance than
    /// the `O3` optimization level and results in very fast compilation.
    pub inline_asm: InlineASM,
    /// Custom header to include in the generated code.
    /// This can be used to include additional libraries or custom functions.
    pub custom_header: Option<String>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            include_header: true,
            inline_asm: InlineASM::default(),
            custom_header: None,
        }
    }
}

impl<T: ExportNumber + SingleFloat> ExpressionEvaluator<T> {
    /// Create a C++ code representation of the evaluation tree.
    /// The resulting source code can be compiled and loaded.
    ///
    /// You can also call `export_cpp` with types [f64], [wide::f64x4] for SIMD, [Complex] over [f64] and [wide::f64x4] for Complex SIMD, and [CudaRealf64] or
    /// [CudaComplexf64] for CUDA output.
    ///
    /// # Examples
    ///
    /// Create a C++ library that evaluates the function `x + y` for `f64` inputs:
    /// ```rust
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::evaluate::{CompiledNumber, FunctionMap, OptimizationSettings};
    /// let fn_map = FunctionMap::new();
    /// let params = vec![parse!("x"), parse!("y")];
    /// let optimization_settings = OptimizationSettings::default();
    /// let evaluator = parse!("x + y")
    ///     .evaluator(&fn_map, &params, optimization_settings)
    ///     .unwrap()
    ///     .map_coeff(&|x| x.to_real().unwrap().to_f64());
    ///
    /// let code = evaluator.export_cpp::<f64>("output.cpp", "my_function", Default::default()).unwrap();
    /// let lib = code.compile("out.so", f64::get_default_compile_options()).unwrap();
    /// let mut compiled_eval = lib.load().unwrap();
    ///
    /// let mut res = [0.];
    /// compiled_eval.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    /// ```
    pub fn export_cpp<F: CompiledNumber>(
        &self,
        path: impl AsRef<Path>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<ExportedCode<F>, std::io::Error> {
        let mut filename = path.as_ref().to_path_buf();
        if filename.extension().map(|x| x != ".cpp").unwrap_or(false) {
            filename.set_extension("cpp");
        }

        let mut source_code = format!(
            "// Auto-generated with Symbolica {}\n// Default build instructions: {} {}\n\n",
            env!("CARGO_PKG_VERSION"),
            F::get_default_compile_options().to_string(),
            filename.to_string_lossy(),
        );

        source_code += &self
            .export_cpp_str::<F>(function_name, settings)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        std::fs::write(&filename, source_code)?;
        Ok(ExportedCode::<F> {
            path: filename,
            function_name: function_name.to_string(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Write the evaluation tree to a C++ source string.
    pub fn export_cpp_str<F: CompiledNumber>(
        &self,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        let function_name = F::construct_function_name(function_name);
        F::export_cpp(self, &function_name, settings)
    }

    pub fn export_simd_str(
        &self,
        function_name: &str,
        settings: ExportSettings,
        complex: bool,
        asm: InlineASM,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include \"xsimd/xsimd.hpp\"\n";
        }

        if complex {
            res += "#include <complex>\n";
            res += "using simd = xsimd::batch<std::complex<double>, xsimd::best_arch>;\n";
        } else {
            res += "using simd = xsimd::batch<double, xsimd::best_arch>;\n";
        }

        match asm {
            InlineASM::AVX2 => {
                res += &self.export_external_cpps();

                res += &format!(
                    "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
                    function_name,
                    self.stack.len()
                );

                if complex {
                    res += &format!(
                        "static const simd {}_CONSTANTS_complex[{}] = {{{}}};\n\n",
                        function_name,
                        self.reserved_indices - self.param_count + 2,
                        {
                            let mut nums = (self.param_count..self.reserved_indices)
                                .map(|i| format!("simd({})", self.stack[i].export()))
                                .collect::<Vec<_>>();
                            nums.push("-0.".to_string()); // used for inversion
                            nums.push("1".to_string()); // used for real inversion
                            nums.join(",")
                        }
                    );
                } else {
                    res += &format!(
                        "static const simd {}_CONSTANTS_double[{}] = {{{}}};\n\n",
                        function_name,
                        self.reserved_indices - self.param_count + 1,
                        {
                            let mut nums = (self.param_count..self.reserved_indices)
                                .map(|i| format!("simd({})", self.stack[i].export()))
                                .collect::<Vec<_>>();
                            nums.push("1".to_string()); // used for inversion
                            nums.join(",")
                        }
                    );
                }

                res += &format!(
                    "\nextern \"C\" void {function_name}(simd *params, simd *Z, simd *out) {{\n"
                );

                if complex {
                    self.export_asm_complex_impl(&self.instructions, function_name, asm, &mut res);
                } else {
                    self.export_asm_double_impl(&self.instructions, function_name, asm, &mut res);
                }

                res += "\treturn;\n}\n";
            }
            InlineASM::None => {
                res += &self.export_generic_cpp_str(function_name, &settings, NumberClass::RealF64);

                res += &format!(
                    "\nextern \"C\" {{\n\tvoid {function_name}(simd *params, simd *buffer, simd *out) {{\n\t\t{function_name}_gen(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n"
                );
            }
            _ => panic!("Bad inline ASM option: {:?}", asm),
        }

        res
    }

    pub fn export_cuda_str(
        &self,
        function_name: &str,
        settings: ExportSettings,
        number_class: NumberClass,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <cuda_runtime.h>\n";
            res += "#include <iostream>\n";
            res += "#include <stdio.h>\n";
            if number_class == NumberClass::ComplexF64 {
                res += "#include <cuda/std/complex>\n";
            } else {
                res += "template<typename T> T conj(T a) { return a; }\n";
            }
        };

        res += &format!("#define ERRMSG_LEN {}\n", CUDA_ERRMSG_LEN);

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n\n";
        }

        if number_class == NumberClass::ComplexF64 {
            res += "typedef cuda::std::complex<double> CudaNumber;\n";
            res += "typedef std::complex<double> Number;\n";
        } else if number_class == NumberClass::RealF64 {
            res += "typedef double CudaNumber;\n";
            res += "typedef double Number;\n";
        }
        res += &self.export_external_cpps();

        res += &format!(
            "\n__device__ void {}(CudaNumber* params, CudaNumber* out, size_t index) {{\n",
            function_name
        );

        res += &format!(
            "\tCudaNumber {};\n",
            (0..self.stack.len())
                .map(|x| format!("Z{}", x))
                .collect::<Vec<_>>()
                .join(", ")
        );

        res += &format!("\tint params_offset = index * {};\n", self.param_count);
        res += &format!(
            "\tint out_offset = index * {};\n",
            self.result_indices.len()
        );

        self.export_cpp_impl("params_offset + ", "CudaNumber", false, &mut res);

        for (i, r) in &mut self.result_indices.iter().enumerate() {
            res += &format!("\tout[out_offset + {i}] = ");
            res += &if *r < self.param_count {
                format!("params[params_offset + {r}]")
            } else if *r < self.reserved_indices {
                self.stack[*r].export_wrapped_with("CudaNumber")
            } else {
                format!("Z{r}")
            };

            res += ";\n";
        }

        res += "\treturn;\n}\n";

        res += &format!(
            r#"
struct {name}_EvaluationData {{
    CudaNumber *params;
    CudaNumber *out;
    size_t n; // Number of evaluations
    size_t block_size; // Number of threads per block
    size_t in_dimension = {in_dimension}; // Number of input parameters
    size_t out_dimension = {out_dimension}; // Number of output parameters
    int last_error = 0; // Last error code
    char last_error_msg[ERRMSG_LEN]; // error string buffer
}};

#define gpuErrchk(ans, data, context) gpuAssert((ans), data, __FILE__, __LINE__, context)
inline int gpuAssert(cudaError_t code, {name}_EvaluationData* data, const char *file, int line, const char *context)
{{
   if (code != cudaSuccess)
   {{
       const char* msg = cudaGetErrorString(code);
       if (msg) {{
           snprintf(
               data->last_error_msg,
               ERRMSG_LEN,
               "%s:%d:%s: CUDA error: %s",
                file,
                line,
                context,
                msg
            );
        }} else {{
            snprintf(
                data->last_error_msg,
                ERRMSG_LEN,
                "%s:%d:%s: CUDA error: unkown",
                file,
                line,
                context
            );
        }}
    }}
    // should always be 0
    if (data->last_error != 0) {{
        fprintf(stderr,
                "%s:%d:%s: CUDA fatal: previous error was not resolved",
                file,
                line,
                context
        );
        // flush output
        fflush(stderr);
        // we crash the evaluation since previous failure was not sanitized
        exit(-1);
    }}
    data->last_error = (int)code;
    return data->last_error;
}}



extern "C" {{

{name}_EvaluationData* {name}_init_data(size_t n, size_t block_size) {{
    {name}_EvaluationData* data = ({name}_EvaluationData*)malloc(sizeof({name}_EvaluationData));
    size_t in_dimension = {in_dimension};
    size_t out_dimension = {out_dimension};
    data->n = n;
    data->in_dimension = in_dimension;
    data->out_dimension = out_dimension;
    data->block_size = block_size;
    data->last_error = 0;
    // return data early since second failure => abort/crash code
    if(gpuErrchk(cudaMalloc((void**)&data->params, n*in_dimension * sizeof(CudaNumber)),data, "init_data_params")) return data;
    if(gpuErrchk(cudaMalloc((void**)&data->out, n*out_dimension*sizeof(CudaNumber)),data, "init_data_out")) return data;
    return data;
}}

int {name}_destroy_data({name}_EvaluationData* data) {{
    // since we free the evaluationData no error can be returned through it
    // neither a Result<(),String> return would make sense in rust drop
    cudaError_t error;
    error = cudaFree(data->params);
    if (error != cudaSuccess) return (int)error;
    error = cudaFree(data->out);
    if (error != cudaSuccess) return (int)error;
    free(data);
    return 0;
}}
}}
       "#,
            name = function_name,
            in_dimension = self.param_count,
            out_dimension = self.result_indices.len()
        );

        res += &format!(
            r#"
extern "C" {{
    __global__ void {name}_cuda(CudaNumber *params, CudaNumber *out, size_t n) {{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < n) {name}(params, out, index);
        return;
    }}
}}
"#,
            name = function_name
        );

        res += &format!(
            r#"
extern "C" {{
    void {name}_vec(Number *params, Number *out, {name}_EvaluationData* data) {{
        size_t n = data->n;
        size_t in_dimension = {in_dimension};
        size_t out_dimension = {out_dimension};

        if(gpuErrchk(cudaMemcpy(data->params, params, n*in_dimension * sizeof(CudaNumber), cudaMemcpyHostToDevice),data, "copy_data_params")) return;

        int blockSize = data->block_size; // Number of threads per block
        int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks
        {name}_cuda<<<gridSize,blockSize>>>(data->params, data->out,n);
        // Collect launch errors
        if(gpuErrchk(cudaPeekAtLastError(), data, "launch")) return;
        // Collect runtime errors
        if(gpuErrchk(cudaDeviceSynchronize(), data, "runtime")) return;

        if(gpuErrchk(cudaMemcpy(out, data->out, n*out_dimension*sizeof(CudaNumber), cudaMemcpyDeviceToHost),data, "copy_data_out")) return;
        return;
    }}
}}
"#,
            name = function_name,
            in_dimension = self.param_count,
            out_dimension = self.result_indices.len()
        );

        res
    }

    pub(super) fn export_generic_cpp_str(
        &self,
        function_name: &str,
        settings: &ExportSettings,
        number_class: NumberClass,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <iostream>\n#include <cmath>\n\n";
            if number_class == NumberClass::ComplexF64 {
                res += "#include <complex>\n";
            } else {
                res += "template<typename T> T conj(T a) { return a; }\n";
            }
        };

        if number_class == NumberClass::ComplexF64 {
            res += "typedef std::complex<double> Number;\n";
        } else if number_class == NumberClass::RealF64 {
            res += "typedef double Number;\n";
        }

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n";
        }
        res += &self.export_external_cpps();

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "\ntemplate<typename T>\nvoid {function_name}_gen(T* params, T* Z, T* out) {{\n"
        );

        self.export_cpp_impl("", "T", true, &mut res);

        for (i, r) in &mut self.result_indices.iter().enumerate() {
            res += &format!("\tout[{i}] = ");
            res += &if *r < self.param_count {
                format!("params[{r}]")
            } else if *r < self.reserved_indices {
                self.stack[*r].export_wrapped_with("T")
            } else {
                format!("Z[{r}]")
            };

            res += ";\n";
        }

        res += "\treturn;\n}\n";

        // if there are non-reals we can not use double evaluation
        assert!(
            !(!self.stack.iter().all(|x| x.is_real()) && number_class == NumberClass::RealF64),
            "Cannot export complex function with real numbers"
        );

        res
    }

    fn export_cpp_impl(
        &self,
        param_offset: &str,
        number_wrapper: &str,
        tmp_array: bool,
        out: &mut String,
    ) {
        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}{}]", param_offset, $i)
                } else if $i < self.reserved_indices {
                    self.stack[$i].export_wrapped_with(number_wrapper)
                } else {
                    // TODO: subtract reserved indices
                    if tmp_array {
                        format!("Z[{}]", $i)
                    } else {
                        format!("Z{}", $i)
                    }
                }
            };
        }

        macro_rules! get_output {
            ($i:expr) => {
                if tmp_array {
                    format!("Z[{}]", $i)
                } else {
                    format!("Z{}", $i)
                }
            };
        }

        let mut close_else_branch = 0;
        for (ins, _c) in &self.instructions {
            match ins {
                Instr::Add(o, a) => {
                    let args = a
                        .iter()
                        .map(|x| get_input!(*x))
                        .collect::<Vec<_>>()
                        .join("+");

                    *out += format!("\t{} = {args};\n", get_output!(o)).as_str();
                }
                Instr::Mul(o, a) => {
                    let args = a
                        .iter()
                        .map(|x| get_input!(*x))
                        .collect::<Vec<_>>()
                        .join("*");

                    *out += format!("\t{} = {args};\n", get_output!(o)).as_str();
                }
                Instr::Pow(o, b, e) => {
                    let base = get_input!(*b);
                    if *e == -1 {
                        *out += format!("\t{} = {number_wrapper}(1) / {base};\n", get_output!(o))
                            .as_str();
                    } else {
                        *out += format!("\t{} = pow({base}, {e});\n", get_output!(o)).as_str();
                    }
                }
                Instr::Powf(o, b, e) => {
                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\t{} = pow({base}, {exp});\n", get_output!(o)).as_str();
                }
                Instr::BuiltinFun(o, s, a) => match s.get_id() {
                    Symbol::EXP_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = exp({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::LOG_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = log({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::SIN_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = sin({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::COS_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = cos({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::SQRT_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = sqrt({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::ABS_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = std::abs({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::CONJ_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = conj({arg});\n", get_output!(o)).as_str();
                    }
                    _ => unreachable!(),
                },
                Instr::ExternalFun(o, s, a) => {
                    let name = &self.external_fns[*s];
                    let args = a.iter().map(|x| get_input!(*x)).collect::<Vec<_>>();

                    *out +=
                        format!("\t{} = {}({});\n", get_output!(o), name, args.join(", ")).as_str();
                }
                Instr::IfElse(cond, _label) => {
                    *out += &format!("\tif ({} != 0.) {{\n", get_input!(*cond));
                }
                Instr::Goto(..) => {
                    *out += "\t} else {\n";
                    close_else_branch += 1;
                }
                Instr::Label(..) => {}
                Instr::Join(o, cond, a, b) => {
                    if close_else_branch > 0 {
                        close_else_branch -= 1;
                        *out += "\t}\n";
                    }
                    let arg_a = get_input!(*a);
                    let arg_b = get_input!(*b);
                    *out += format!(
                        "\t{} = ({} != 0.) ? {} : {};\n",
                        get_output!(o),
                        get_input!(*cond),
                        arg_a,
                        arg_b
                    )
                    .as_str();
                }
            }
        }
    }

    pub(super) fn export_asm_real_str(
        &self,
        function_name: &str,
        settings: &ExportSettings,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <iostream>\n#include <cmath>\n\n#include <complex>\n";
        };

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n";
        }
        res += &self.export_external_cpps();

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        if self.stack.iter().all(|x| x.is_real()) {
            res += &format!(
                "static const double {}_CONSTANTS_double[{}] = {{{}}};\n\n",
                function_name,
                self.reserved_indices - self.param_count + 1,
                {
                    let mut nums = (self.param_count..self.reserved_indices)
                        .map(|i| format!("double({})", self.stack[i].export()))
                        .collect::<Vec<_>>();
                    nums.push("1".to_string()); // used for inversion
                    nums.join(",")
                }
            );

            res += &format!(
                "extern \"C\" void {function_name}(const double *params, double* Z, double *out)\n{{\n"
            );

            self.export_asm_double_impl(
                &self.instructions,
                function_name,
                settings.inline_asm,
                &mut res,
            );

            res += "\treturn;\n}\n";
        } else {
            res += &format!(
                "extern \"C\" void {function_name}(const double *params, double* Z, double *out)\n{{\n\tstd::cout << \"Cannot evaluate complex function with doubles\" << std::endl;\n\treturn; \n}}",
            );
        }
        res
    }

    pub(super) fn export_asm_complex_str(
        &self,
        function_name: &str,
        settings: &ExportSettings,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <iostream>\n#include <complex>\n#include <cmath>\n\n";
        };

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n";
        }
        res += &self.export_external_cpps();

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "static const std::complex<double> {}_CONSTANTS_complex[{}] = {{{}}};\n\n",
            function_name,
            self.reserved_indices - self.param_count + 2,
            {
                let mut nums = (self.param_count..self.reserved_indices)
                    .map(|i| format!("std::complex<double>({})", self.stack[i].export()))
                    .collect::<Vec<_>>();
                nums.push("std::complex<double>(0, -0.)".to_string()); // used for complex inversion
                nums.push("1".to_string()); // used for real inversion
                nums.join(",")
            }
        );

        res += &format!(
            "extern \"C\" void {function_name}(const std::complex<double> *params, std::complex<double> *Z, std::complex<double> *out)\n{{\n"
        );

        self.export_asm_complex_impl(
            &self.instructions,
            function_name,
            settings.inline_asm,
            &mut res,
        );

        res + "\treturn;\n}\n\n"
    }

    fn export_asm_double_impl(
        &self,
        instr: &[(Instr, ComplexPhase)],
        function_name: &str,
        asm_flavour: InlineASM,
        out: &mut String,
    ) -> bool {
        let mut second_index = 0;

        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!(
                        "{}_CONSTANTS_double[{}]",
                        function_name,
                        $i - self.param_count
                    )
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! asm_load {
            ($i:expr) => {
                match asm_flavour {
                    InlineASM::X64 => {
                        if $i < self.param_count {
                            format!("{}(%2)", $i * 8)
                        } else if $i < self.reserved_indices {
                            format!("{}(%1)", ($i - self.param_count) * 8)
                        } else {
                            // TODO: subtract reserved indices
                            format!("{}(%0)", $i * 8)
                        }
                    }
                    InlineASM::AVX2 => {
                        if $i < self.param_count {
                            format!("{}(%2)", $i * 32)
                        } else if $i < self.reserved_indices {
                            format!("{}(%1)", ($i - self.param_count) * 32)
                        } else {
                            // TODO: subtract reserved indices
                            format!("{}(%0)", $i * 32)
                        }
                    }
                    InlineASM::AArch64 => {
                        if $i < self.param_count {
                            let dest = $i * 8;

                            if dest > 32760 {
                                // maximum allowed shift is 12 bits
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %2, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                format!("[x8, {}]", rest)
                            } else {
                                format!("[%2, {}]", dest)
                            }
                        } else if $i < self.reserved_indices {
                            let dest = ($i - self.param_count) * 8;
                            if dest > 32760 {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %1, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                format!("[x8, {}]", rest)
                            } else {
                                format!("[%1, {}]", dest)
                            }
                        } else {
                            // TODO: subtract reserved indices
                            let dest = $i * 8;
                            if dest > 32760 && (dest < second_index || dest > 32760 + second_index)
                            {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                second_index = coeff << shift;
                                let rest = dest - second_index;
                                *out += &format!(
                                    "\t\t\"add x8, %0, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                format!("[x8, {}]", rest)
                            } else if dest <= 32760 {
                                format!("[%0, {}]", dest)
                            } else {
                                let offset = dest - second_index;
                                format!("[x8, {}]", offset)
                            }
                        }
                    }
                    InlineASM::None => unreachable!(),
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n",  function_name);
                        }
                        InlineASM::AVX2 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n",  function_name);
                        }
                        InlineASM::AArch64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n",  function_name);
                            #[allow(unused_assignments)] { second_index = 0;}; // the second index in x8 will be lost after the block, so reset it
                        }
                        InlineASM::None => unreachable!(),
                    }
                    $in_block = false;
                }
            };
        }

        let mut reg_last_use = vec![self.instructions.len(); self.instructions.len()];
        let mut stack_to_reg = HashMap::default();

        for (i, (ins, _)) in instr.iter().enumerate() {
            match ins {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    for x in a {
                        if x >= &self.reserved_indices {
                            reg_last_use[stack_to_reg[x]] = i;
                        }
                    }

                    stack_to_reg.insert(r, i);
                }
                Instr::Pow(r, b, _) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::Powf(r, b, e) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    if e >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[e]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::BuiltinFun(r, _, b) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::IfElse(c, _) => {
                    if c >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[c]] = i;
                    }
                }
                Instr::Join(r, c, t, f) => {
                    if c >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[c]] = i;
                    }
                    if t >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[t]] = i;
                    }
                    if f >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[f]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::Goto(..) | Instr::Label(..) => {}
            }
        }

        for x in &self.result_indices {
            if x >= &self.reserved_indices {
                reg_last_use[stack_to_reg[x]] = self.instructions.len();
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum MemOrReg {
            Mem(usize),
            Reg(usize),
        }

        #[derive(Debug, Clone)]
        #[allow(dead_code)]
        enum RegInstr {
            Add(MemOrReg, u16, Vec<MemOrReg>),
            Mul(MemOrReg, u16, Vec<MemOrReg>),
            Pow(MemOrReg, u16, MemOrReg, i64),
            Sqrt(MemOrReg, u16, MemOrReg),
            Powf(usize, usize, usize),
            BuiltinFun(usize, Symbol, usize),
            ExternalFun(usize, usize, Vec<usize>),
            IfElse(usize),
            Goto,
            Label(Label),
            Join(usize, usize, usize, usize),
        }

        let mut new_instr: Vec<RegInstr> = instr
            .iter()
            .map(|(i, _)| match i {
                Instr::Add(r, a) => RegInstr::Add(
                    MemOrReg::Mem(*r),
                    u16::MAX,
                    a.iter().map(|x| MemOrReg::Mem(*x)).collect(),
                ),
                Instr::Mul(r, a) => RegInstr::Mul(
                    MemOrReg::Mem(*r),
                    u16::MAX,
                    a.iter().map(|x| MemOrReg::Mem(*x)).collect(),
                ),
                Instr::Pow(r, b, e) => {
                    RegInstr::Pow(MemOrReg::Mem(*r), u16::MAX, MemOrReg::Mem(*b), *e)
                }
                Instr::Powf(r, b, e) => RegInstr::Powf(*r, *b, *e),
                Instr::BuiltinFun(r, s, a) => {
                    if *s == Symbol::SQRT {
                        RegInstr::Sqrt(MemOrReg::Mem(*r), u16::MAX, MemOrReg::Mem(*a))
                    } else {
                        RegInstr::BuiltinFun(*r, *s, *a)
                    }
                }
                Instr::ExternalFun(r, s, a) => RegInstr::ExternalFun(*r, *s, a.clone()),
                Instr::IfElse(c, _) => RegInstr::IfElse(*c),
                Instr::Goto(_) => RegInstr::Goto,
                Instr::Label(l) => RegInstr::Label(*l),
                Instr::Join(r, c, a, b) => RegInstr::Join(*r, *c, *a, *b),
            })
            .collect();

        // sort the list of instructions based on the distance
        let mut reg_list = reg_last_use.iter().enumerate().collect::<Vec<_>>();
        reg_list.sort_by_key(|x| (*x.1 - x.0, x.0));

        'next: for (j, last_use) in reg_list {
            if *last_use == self.instructions.len() {
                continue;
            }

            let old_reg = if let RegInstr::Add(r, _, _)
            | RegInstr::Mul(r, _, _)
            | RegInstr::Pow(r, _, _, -1) = &new_instr[j]
            {
                if let MemOrReg::Mem(r) = r {
                    *r
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // find free registers in the range
            // start at j+1 as we can recycle registers that are last used in iteration j
            let mut free_regs = u16::MAX & !(1 << 15); // leave xmmm15 open

            for k in &new_instr[j + 1..=*last_use] {
                match k {
                    RegInstr::Add(_, f, _)
                    | RegInstr::Mul(_, f, _)
                    | RegInstr::Pow(_, f, _, -1) => {
                        free_regs &= f;
                    }

                    _ => {
                        free_regs = 0; // the current instruction is not allowed to be used outside of ASM blocks
                    }
                }

                if free_regs == 0 {
                    continue 'next;
                }
            }

            if let Some(k) = (0..16).position(|k| free_regs & (1 << k) != 0) {
                if let RegInstr::Add(r, _, _) | RegInstr::Mul(r, _, _) | RegInstr::Pow(r, _, _, _) =
                    &mut new_instr[j]
                {
                    *r = MemOrReg::Reg(k);
                }

                for l in &mut new_instr[j + 1..=*last_use] {
                    match l {
                        RegInstr::Add(_, f, a) | RegInstr::Mul(_, f, a) => {
                            *f &= !(1 << k); // FIXME: do not set on last use?
                            for x in a {
                                if *x == MemOrReg::Mem(old_reg) {
                                    *x = MemOrReg::Reg(k);
                                }
                            }
                        }
                        RegInstr::Pow(_, f, a, -1) | RegInstr::Sqrt(_, f, a) => {
                            *f &= !(1 << k); // FIXME: do not set on last use?
                            if *a == MemOrReg::Mem(old_reg) {
                                *a = MemOrReg::Reg(k);
                            }
                        }
                        RegInstr::Pow(_, _, _, _) => {
                            panic!("use outside of ASM block");
                        }
                        RegInstr::Powf(_, a, b) => {
                            if *a == old_reg {
                                panic!("use outside of ASM block");
                            }
                            if *b == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::BuiltinFun(_, _, a) => {
                            if *a == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::ExternalFun(_, _, a) => {
                            if a.contains(&old_reg) {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::IfElse(c) => {
                            if *c == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::Label(_) => {}
                        RegInstr::Goto => {}
                        RegInstr::Join(_, c, a, b) => {
                            if *c == old_reg || *a == old_reg || *b == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                    }
                }

                // TODO: if last use is not already set to a register, we can set it to the current one
                // this prevents a copy
            }
        }

        let mut label_stack = vec![];
        let mut label_join_info = HashMap::default();
        let mut in_join_section = false;
        for (ins, _) in instr {
            if in_join_section && !matches!(ins, Instr::Join(..)) {
                in_join_section = false;
                label_stack.pop().unwrap();
            }

            match ins {
                Instr::IfElse(_, label) => {
                    label_stack.push((*label, None));
                }
                Instr::Goto(l) => {
                    if let Some(last) = label_stack.last_mut() {
                        last.1 = Some(*l);
                    }
                }
                Instr::Join(o, _, a, b) => {
                    in_join_section = true; // could be more than one join if vectorized

                    if let Some((label, label_2)) = label_stack.last() {
                        label_join_info
                            .entry(*label)
                            .or_insert(vec![])
                            .push((*o, *a));
                        label_join_info
                            .entry(label_2.unwrap())
                            .or_insert(vec![])
                            .push((*o, *b));
                    } else {
                        unreachable!("Goto without matching IfElse");
                    }
                }
                _ => {
                    in_join_section = false;
                }
            }
        }

        let mut in_asm_block = false;
        let mut next_label_is_true_branch_end = false;
        for ins in &new_instr {
            match ins {
                RegInstr::Add(o, free, a) | RegInstr::Mul(o, free, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    let oper = if matches!(ins, RegInstr::Add(_, _, _)) {
                        "add"
                    } else {
                        "mul"
                    };

                    match o {
                        MemOrReg::Reg(out_reg) => {
                            if let Some(j) = a.iter().find(|x| **x == MemOrReg::Reg(*out_reg)) {
                                // we can recycle the register completely
                                let mut first_skipped = false;
                                for i in a {
                                    if first_skipped || i != j {
                                        match i {
                                            MemOrReg::Reg(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd %%xmm{k}, %%xmm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd %%ymm{k}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d{k}, d{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                            MemOrReg::Mem(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                    );

                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                        }
                                    }
                                    first_skipped |= i == j;
                                }
                            } else if let Some(MemOrReg::Reg(j)) =
                                a.iter().find(|x| matches!(x, MemOrReg::Reg(_)))
                            {
                                match asm_flavour {
                                    InlineASM::X64 => {
                                        *out += &format!(
                                            "\t\t\"movapd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AVX2 => {
                                        *out += &format!(
                                            "\t\t\"vmovapd %%ymm{j}, %%ymm{out_reg}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AArch64 => {
                                        *out += &format!("\t\t\"fmov d{out_reg}, d{j}\\n\\t\"\n");
                                    }
                                    InlineASM::None => unreachable!(),
                                }

                                let mut first_skipped = false;
                                for i in a {
                                    if first_skipped || *i != MemOrReg::Reg(*j) {
                                        match i {
                                            MemOrReg::Reg(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd %%xmm{k}, %%xmm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd %%ymm{k}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d{k}, d{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                            MemOrReg::Mem(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                    );

                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                        }
                                    }
                                    first_skipped |= *i == MemOrReg::Reg(*j);
                                }
                            } else {
                                if let MemOrReg::Mem(k) = &a[0] {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            let addr = asm_load!(*k);
                                            *out += &format!(
                                                "\t\t\"movsd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            let addr = asm_load!(*k);
                                            *out += &format!(
                                                "\t\t\"vmovapd {addr}, %%ymm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            let addr = asm_load!(*k);
                                            *out +=
                                                &format!("\t\t\"ldr d{out_reg}, {addr}\\n\\t\"\n",);
                                        }
                                        InlineASM::None => unreachable!(),
                                    }
                                } else {
                                    unreachable!();
                                }

                                for i in &a[1..] {
                                    if let MemOrReg::Mem(k) = i {
                                        match asm_flavour {
                                            InlineASM::X64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                let addr = asm_load!(*k);
                                                *out +=
                                                    &format!("\t\t\"ldr d31, {addr}\\n\\t\"\n",);

                                                *out += &format!(
                                                    "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        }
                                    }
                                }
                            }
                        }
                        MemOrReg::Mem(out_mem) => {
                            // TODO: we would like a last-use check of the free here. Now we need to move
                            if let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) {
                                if let Some(MemOrReg::Reg(j)) =
                                    a.iter().find(|x| matches!(x, MemOrReg::Reg(_)))
                                {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movapd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovapd %%ymm{j}, %%ymm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out +=
                                                &format!("\t\t\"fmov d{out_reg}, d{j}\\n\\t\"\n");
                                        }
                                        InlineASM::None => unreachable!(),
                                    }

                                    let mut first_skipped = false;
                                    for i in a {
                                        if first_skipped || *i != MemOrReg::Reg(*j) {
                                            match i {
                                                MemOrReg::Reg(k) => match asm_flavour {
                                                    InlineASM::X64 => {
                                                        *out += &format!(
                                                            "\t\t\"{oper}sd %%xmm{k}, %%xmm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AVX2 => {
                                                        *out += &format!(
                                                            "\t\t\"v{oper}pd %%ymm{k}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AArch64 => {
                                                        *out += &format!(
                                                            "\t\t\"f{oper} d{out_reg}, d{k}, d{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::None => unreachable!(),
                                                },
                                                MemOrReg::Mem(k) => match asm_flavour {
                                                    InlineASM::X64 => {
                                                        let addr = asm_load!(*k);
                                                        *out += &format!(
                                                            "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AVX2 => {
                                                        let addr = asm_load!(*k);
                                                        *out += &format!(
                                                            "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AArch64 => {
                                                        let addr = asm_load!(*k);
                                                        *out += &format!(
                                                            "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                        );

                                                        *out += &format!(
                                                            "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::None => unreachable!(),
                                                },
                                            }
                                        }

                                        first_skipped |= *i == MemOrReg::Reg(*j);
                                    }
                                } else {
                                    if let MemOrReg::Mem(k) = &a[0] {
                                        let addr = asm_load!(*k);
                                        match asm_flavour {
                                            InlineASM::X64 => {
                                                *out += &format!(
                                                    "\t\t\"movsd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                *out += &format!(
                                                    "\t\t\"vmovapd {addr}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                *out += &format!(
                                                    "\t\t\"ldr d{out_reg}, {addr}\\n\\t\"\n",
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        }
                                    } else {
                                        unreachable!();
                                    }

                                    for i in &a[1..] {
                                        if let MemOrReg::Mem(k) = i {
                                            let addr = asm_load!(*k);
                                            match asm_flavour {
                                                InlineASM::X64 => {
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    *out += &format!(
                                                        "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                    );

                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            }
                                        }
                                    }
                                }

                                let addr = asm_load!(*out_mem);
                                match asm_flavour {
                                    InlineASM::X64 => {
                                        *out += &format!(
                                            "\t\t\"movsd %%xmm{out_reg}, {addr}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AVX2 => {
                                        *out += &format!(
                                            "\t\t\"vmovupd %%ymm{out_reg}, {addr}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AArch64 => {
                                        *out += &format!("\t\t\"str d{out_reg}, {addr}\\n\\t\"\n",);
                                    }
                                    InlineASM::None => unreachable!(),
                                }
                            } else {
                                unreachable!("No free registers");
                                // move the value of xmm0 into the memory location of the output register
                                // and then swap later?
                            }
                        }
                    }
                }
                RegInstr::Pow(o, free, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        match o {
                            MemOrReg::Reg(out_reg) => {
                                if *b == MemOrReg::Reg(*out_reg) {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            if let Some(tmp_reg) =
                                                (0..16).position(|k| free & (1 << k) != 0)
                                            {
                                                *out += &format!(
                                                    "\t\t\"movapd %%xmm{out_reg}, %%xmm{tmp_reg}\\n\\t\"\n"
                                                );

                                                *out += &format!(
                                                    "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                                    (self.reserved_indices - self.param_count) * 8,
                                                    out_reg
                                                );

                                                *out += &format!(
                                                    "\t\t\"divsd %%xmm{tmp_reg}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            } else {
                                                panic!("No free registers for division")
                                            }
                                        }
                                        InlineASM::AVX2 => {
                                            if let Some(tmp_reg) =
                                                (0..16).position(|k| free & (1 << k) != 0)
                                            {
                                                *out += &format!(
                                                    "\t\t\"vmovapd %%ymm{out_reg}, %%ymm{tmp_reg}\\n\\t\"\n"
                                                );

                                                *out += &format!(
                                                    "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                                                    (self.reserved_indices - self.param_count) * 32,
                                                    out_reg
                                                );

                                                *out += &format!(
                                                    "\t\t\"vdivsd %%ymm{tmp_reg}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            } else {
                                                panic!("No free registers for division")
                                            }
                                        }
                                        InlineASM::AArch64 => {
                                            *out += &format!(
                                                "\t\t\"ldr d31, [%1, {}]\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 8
                                            );
                                            *out += &format!(
                                                "\t\t\"fdiv d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::None => unreachable!(),
                                    }
                                } else {
                                    // load 1 into out_reg
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 8,
                                                out_reg,
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 32,
                                                out_reg,
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out += &format!(
                                                "\t\t\"ldr d{}, [%1, {}]\\n\\t\"\n",
                                                out_reg,
                                                (self.reserved_indices - self.param_count) * 8
                                            );
                                        }
                                        InlineASM::None => unreachable!(),
                                    }

                                    match b {
                                        MemOrReg::Reg(j) => match asm_flavour {
                                            InlineASM::X64 => {
                                                *out += &format!(
                                                    "\t\t\"divsd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                *out += &format!(
                                                    "\t\t\"vdivpd %%ymm{j}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d{j}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                        MemOrReg::Mem(k) => match asm_flavour {
                                            InlineASM::X64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"divsd {addr}, %%xmm{out_reg}\\n\\t\"\n",
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"vdivpd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n",
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!("\t\t\"ldr d31, {addr}\\n\\t\"\n");

                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d31\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                    }
                                }
                            }
                            MemOrReg::Mem(out_mem) => {
                                if let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 8,
                                                out_reg
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 32,
                                                out_reg
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out += &format!(
                                                "\t\t\"ldr d{}, [%1, {}]\\n\\t\"\n",
                                                out_reg,
                                                (self.reserved_indices - self.param_count) * 8
                                            );
                                        }
                                        InlineASM::None => unreachable!(),
                                    }

                                    match b {
                                        MemOrReg::Reg(j) => match asm_flavour {
                                            InlineASM::X64 => {
                                                *out += &format!(
                                                    "\t\t\"divsd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                *out += &format!(
                                                    "\t\t\"vdivpd %%ymm{j}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d{j}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                        MemOrReg::Mem(k) => match asm_flavour {
                                            InlineASM::X64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"divsd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"vdivpd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                let addr = asm_load!(*k);
                                                *out +=
                                                    &format!("\t\t\"ldr d31, {addr}\\n\\t\"\n",);

                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d31\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                    }

                                    let addr = asm_load!(*out_mem);
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movsd %%xmm{out_reg}, {addr}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovupd %%ymm{out_reg}, {addr}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out +=
                                                &format!("\t\t\"str d{out_reg}, {addr}\\n\\t\"\n",);
                                        }
                                        InlineASM::None => unreachable!(),
                                    }
                                } else {
                                    unreachable!("No free registers");
                                    // move the value of xmm0 into the memory location of the output register
                                    // and then swap later?
                                }
                            }
                        }
                    } else {
                        unreachable!(
                            "Powers other than -1 should have been removed at an earlier stage"
                        );
                    }
                }
                RegInstr::Sqrt(o, free, b) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    let out_reg = match o {
                        MemOrReg::Reg(out_reg) => *out_reg,
                        MemOrReg::Mem(_) => {
                            let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) else {
                                unreachable!("No free registers for sqrt output")
                            };
                            out_reg
                        }
                    };

                    let b_reg = match b {
                        MemOrReg::Reg(b_reg) => *b_reg,
                        MemOrReg::Mem(k) => {
                            match asm_flavour {
                                InlineASM::X64 => {
                                    let addr = asm_load!(*k);
                                    *out +=
                                        &format!("\t\t\"movsd {addr}, %%xmm{out_reg}\\n\\t\"\n",);
                                }
                                InlineASM::AVX2 => {
                                    let addr = asm_load!(*k);
                                    *out +=
                                        &format!("\t\t\"vmovupd {addr}, %%ymm{out_reg}\\n\\t\"\n",);
                                }
                                InlineASM::AArch64 => {
                                    let addr = asm_load!(*k);
                                    *out += &format!("\t\t\"ldr d{out_reg}, {addr}\\n\\t\"\n",);
                                }
                                InlineASM::None => unreachable!(),
                            }

                            out_reg
                        }
                    };

                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t\"sqrtsd %%xmm{b_reg}, %%xmm{out_reg}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            *out +=
                                &format!("\t\t\"vsqrtpd %%ymm{b_reg}, %%ymm{out_reg}\\n\\t\"\n");
                        }
                        InlineASM::AArch64 => {
                            *out += &format!("\t\t\"fsqrt d{out_reg}, d{b_reg}\\n\\t\"\n");
                        }
                        InlineASM::None => unreachable!(),
                    }

                    if let MemOrReg::Mem(out_mem) = o {
                        let out_mem = asm_load!(*out_mem);
                        match asm_flavour {
                            InlineASM::X64 => {
                                *out += &format!("\t\t\"movsd %%xmm{out_reg}, {out_mem}\\n\\t\"\n");
                            }
                            InlineASM::AVX2 => {
                                *out +=
                                    &format!("\t\t\"vmovupd %%ymm{out_reg}, {out_mem}\\n\\t\"\n");
                            }
                            InlineASM::AArch64 => {
                                *out += &format!("\t\t\"str d{out_reg}, {out_mem}\\n\\t\"\n",);
                            }
                            InlineASM::None => unreachable!(),
                        }
                    }
                }
                RegInstr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);

                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\tZ[{o}] = pow({base}, {exp});\n").as_str();
                }
                RegInstr::BuiltinFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let arg = get_input!(*a);

                    match s.get_id() {
                        Symbol::EXP_ID => {
                            *out += format!("\tZ[{o}] = exp({arg});\n").as_str();
                        }
                        Symbol::LOG_ID => {
                            *out += format!("\tZ[{o}] = log({arg});\n").as_str();
                        }
                        Symbol::SIN_ID => {
                            *out += format!("\tZ[{o}] = sin({arg});\n").as_str();
                        }
                        Symbol::COS_ID => {
                            *out += format!("\tZ[{o}] = cos({arg});\n").as_str();
                        }
                        Symbol::ABS_ID => {
                            *out += format!("\tZ[{o}] = std::abs({arg});\n").as_str();
                        }
                        Symbol::CONJ_ID => {
                            *out += format!("\tZ[{o}] = {arg};\n").as_str();
                        }
                        _ => unreachable!(),
                    }
                }
                RegInstr::ExternalFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let name = &self.external_fns[*s];
                    let args = a.iter().map(|x| get_input!(*x)).collect::<Vec<_>>();

                    *out += format!("\tZ[{}] = {}({});\n", o, name, args.join(", ")).as_str();
                }
                RegInstr::IfElse(cond) => {
                    end_asm_block!(in_asm_block);

                    if asm_flavour == InlineASM::AVX2 {
                        *out += &format!("\tif (all({} != 0.)) {{\n", get_input!(*cond));
                    } else {
                        *out += &format!("\tif ({} != 0.) {{\n", get_input!(*cond));
                    }
                }
                RegInstr::Goto => {
                    next_label_is_true_branch_end = true;
                }
                RegInstr::Label(l) => {
                    end_asm_block!(in_asm_block);

                    for (o, b) in label_join_info.get(l).unwrap() {
                        let arg_a = get_input!(*o);
                        let arg_b = get_input!(*b);
                        *out += &format!("\t{} = {};\n", arg_a, arg_b);
                    }

                    if next_label_is_true_branch_end {
                        *out += "\t} else {\n";
                        next_label_is_true_branch_end = false;
                    } else {
                        *out += "\t}\n";
                    }
                }
                RegInstr::Join(_, _, _, _) => {}
            }
        }

        end_asm_block!(in_asm_block);

        let mut regcount = 0;
        *out += "\t__asm__(\n";
        for (i, r) in self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movsd {}(%2), %%xmm{}\\n\\t\"\n", r * 8, regcount);
                    }
                    InlineASM::AVX2 => {
                        *out +=
                            &format!("\t\t\"vmovupd {}(%2), %%ymm{}\\n\\t\"\n", r * 32, regcount);
                    }
                    InlineASM::AArch64 => {
                        let addr = asm_load!(*r);
                        *out += &format!("\t\t\"ldr d{}, {}\\n\\t\"\n", regcount, addr);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else if *r < self.reserved_indices {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!(
                            "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                            (r - self.param_count) * 8,
                            regcount
                        );
                    }
                    InlineASM::AVX2 => {
                        *out += &format!(
                            "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                            (r - self.param_count) * 32,
                            regcount
                        );
                    }
                    InlineASM::AArch64 => {
                        let addr = asm_load!(*r);
                        *out += &format!("\t\t\"ldr d{}, {}\\n\\t\"\n", regcount, addr);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movsd {}(%0), %%xmm{}\\n\\t\"\n", r * 8, regcount);
                    }
                    InlineASM::AVX2 => {
                        *out +=
                            &format!("\t\t\"vmovupd {}(%0), %%ymm{}\\n\\t\"\n", r * 32, regcount);
                    }
                    InlineASM::AArch64 => {
                        let addr = asm_load!(*r);
                        *out += &format!("\t\t\"ldr d{}, {}\\n\\t\"\n", regcount, addr);
                    }
                    InlineASM::None => unreachable!(),
                }
            }

            match asm_flavour {
                InlineASM::X64 => {
                    *out += &format!("\t\t\"movsd %%xmm{}, {}(%3)\\n\\t\"\n", regcount, i * 8);
                }
                InlineASM::AVX2 => {
                    *out += &format!("\t\t\"vmovupd %%ymm{}, {}(%3)\\n\\t\"\n", regcount, i * 32);
                }
                InlineASM::AArch64 => {
                    let dest = i * 8;
                    if dest > 32760 {
                        let d = dest.ilog2();
                        let shift = d.min(12);
                        let coeff = dest / (1 << shift);
                        let rest = dest - (coeff << shift);
                        second_index = 0;
                        *out += &format!("\t\t\"add x8, %3, {}, lsl {}\\n\\t\"\n", coeff, shift);
                        *out += &format!("\t\t\"str d{}, [x8, {}]\\n\\t\"\n", regcount, rest);
                    } else {
                        *out += &format!("\t\t\"str d{}, [%3, {}]\\n\\t\"\n", regcount, i * 8);
                    }
                }
                InlineASM::None => unreachable!(),
            }
            regcount = (regcount + 1) % 16;
        }

        match asm_flavour {
            InlineASM::X64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n"
                );
            }
            InlineASM::AVX2 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n"
                );
            }
            InlineASM::AArch64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n"
                );
            }
            InlineASM::None => unreachable!(),
        }
        in_asm_block
    }

    fn export_asm_complex_impl(
        &self,
        instr: &[(Instr, ComplexPhase)],
        function_name: &str,
        asm_flavour: InlineASM,
        out: &mut String,
    ) -> bool {
        let mut second_index = 0;

        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!(
                        "{}_CONSTANTS_complex[{}]",
                        function_name,
                        $i - self.param_count
                    )
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! asm_load {
            ($i:expr) => {
                match asm_flavour {
                    InlineASM::X64 => {
                        if $i < self.param_count {
                            (format!("{}(%2)", $i * 16), String::new())
                        } else if $i < self.reserved_indices {
                            (
                                format!("{}(%1)", ($i - self.param_count) * 16),
                                "NA".to_owned(),
                            )
                        } else {
                            // TODO: subtract reserved indices
                            (format!("{}(%0)", $i * 16), String::new())
                        }
                    }
                    InlineASM::AVX2 => {
                        if $i < self.param_count {
                            (format!("{}(%2)", $i * 64), format!("{}(%2)", $i * 64 + 32))
                        } else if $i < self.reserved_indices {
                            (
                                format!("{}(%1)", ($i - self.param_count) * 64),
                                format!("{}(%1)", ($i - self.param_count) * 64 + 32),
                            )
                        } else {
                            // TODO: subtract reserved indices
                            (format!("{}(%0)", $i * 64), format!("{}(%0)", $i * 64 + 32))
                        }
                    }
                    InlineASM::AArch64 => {
                        if $i < self.param_count {
                            let dest = $i * 16;

                            if dest > 32760 {
                                // maximum allowed shift is 12 bits
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %2, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                (format!("[x8, {}]", rest), format!("[x8, {}]", rest + 8))
                            } else {
                                (format!("[%2, {}]", dest), format!("[%2, {}]", dest + 8))
                            }
                        } else if $i < self.reserved_indices {
                            let dest = ($i - self.param_count) * 16;
                            if dest > 32760 {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %1, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                (format!("[x8, {}]", rest), format!("[x8, {}]", rest + 8))
                            } else {
                                (format!("[%1, {}]", dest), format!("[%1, {}]", dest + 8))
                            }
                        } else {
                            // TODO: subtract reserved indices
                            let dest = $i * 16;
                            if dest > 32760 && (dest < second_index || dest > 32760 + second_index)
                            {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                second_index = coeff << shift;
                                let rest = dest - second_index;
                                *out += &format!(
                                    "\t\t\"add x8, %0, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                (format!("[x8, {}]", rest), format!("[x8, {}]", rest + 8))
                            } else if dest <= 32760 {
                                (format!("[%0, {}]", dest), format!("[%0, {}]", dest + 8))
                            } else {
                                let offset = dest - second_index;
                                (format!("[x8, {}]", offset), format!("[x8, {}]", offset + 8))
                            }
                        }
                    }
                    InlineASM::None => unreachable!(),
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n",  function_name);
                        }
                        InlineASM::AVX2 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n",  function_name);
                        }
                        InlineASM::AArch64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n",  function_name);
                            #[allow(unused_assignments)] { second_index = 0;} // the second index in x8 will be lost after the block, so reset it
                        }
                        InlineASM::None => unreachable!(),
                    }
                    $in_block = false;
                }
            };
        }

        let mut label_stack = vec![];
        let mut label_join_info = HashMap::default();
        let mut in_join_section = false;
        for (ins, _) in instr {
            if in_join_section && !matches!(ins, Instr::Join(..)) {
                in_join_section = false;
                label_stack.pop().unwrap();
            }

            match ins {
                Instr::IfElse(_, label) => {
                    label_stack.push((*label, None));
                }
                Instr::Goto(l) => {
                    if let Some(last) = label_stack.last_mut() {
                        last.1 = Some(*l);
                    }
                }
                Instr::Join(o, _, a, b) => {
                    in_join_section = true; // could be more than one join if vectorized

                    if let Some((label, label_2)) = label_stack.last() {
                        label_join_info
                            .entry(*label)
                            .or_insert(vec![])
                            .push((*o, *a));
                        label_join_info
                            .entry(label_2.unwrap())
                            .or_insert(vec![])
                            .push((*o, *b));
                    } else {
                        unreachable!("Goto without matching IfElse");
                    }
                }
                _ => {
                    in_join_section = false;
                }
            }
        }

        let mut in_asm_block = false;
        let mut next_label_is_true_branch_end = false;
        for (ins, c) in instr {
            match ins {
                Instr::Add(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    match asm_flavour {
                        InlineASM::X64 => {
                            let (addr, _) = asm_load!(a[0]);
                            *out += &format!("\t\t\"movupd {addr}, %%xmm0\\n\\t\"\n");

                            for i in &a[1..] {
                                let (addr, _) = asm_load!(*i);
                                *out += &format!("\t\t\"movupd {addr}, %%xmm1\\n\\t\"\n");
                                *out += &format!("\t\t\"addpd %%xmm1, %%xmm0\\n\\t\"\n");
                            }
                            let (addr, _) = asm_load!(*o);
                            *out += &format!("\t\t\"movupd %%xmm0, {addr}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            let (addr, comp_addr) = asm_load!(a[0]);
                            *out += &format!("\t\t\"vmovupd {addr}, %%ymm0\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd {comp_addr}, %%ymm1\\n\\t\"\n");

                            for i in &a[1..] {
                                let (addr, imag_addr) = asm_load!(*i);
                                *out += &format!("\t\t\"vaddpd {addr}, %%ymm0, %%ymm0\\n\\t\"\n");
                                *out +=
                                    &format!("\t\t\"vaddpd {imag_addr}, %%ymm1, %%ymm1\\n\\t\"\n");
                            }
                            let (addr, imag_addr) = asm_load!(*o);
                            *out += &format!("\t\t\"vmovupd %%ymm0, {addr}\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd %%ymm1, {imag_addr}\\n\\t\"\n");
                        }
                        InlineASM::AArch64 => {
                            let (addr, _) = asm_load!(a[0]);
                            *out += &format!("\t\t\"ldr q0, {addr}\\n\\t\"\n");

                            for i in &a[1..] {
                                let (addr, _) = asm_load!(*i);
                                *out += &format!("\t\t\"ldr q1, {addr}\\n\\t\"\n");
                                *out += "\t\t\"fadd v0.2d, v1.2d, v0.2d\\n\\t\"\n";
                            }

                            let (addr, _) = asm_load!(*o);
                            *out += &format!("\t\t\"str q0, {addr}\\n\\t\"\n");
                        }
                        InlineASM::None => unreachable!(),
                    }
                }
                Instr::Mul(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    macro_rules! load_complex {
                        ($i: expr, $r: expr) => {
                            let (addr_re, addr_im) = asm_load!($r);
                            match asm_flavour {
                                InlineASM::X64 => {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm{}\\n\\t\"\n",
                                        addr_re,
                                        $i + 1,
                                    );
                                }
                                InlineASM::AVX2 => {
                                    *out += &format!(
                                        "\t\t\"vmovupd {}, %%ymm{}\\n\\t\"\n",
                                        addr_re,
                                        2 * $i,
                                    );
                                    *out += &format!(
                                        "\t\t\"vmovupd {}, %%ymm{}\\n\\t\"\n",
                                        addr_im,
                                        2 * $i + 1,
                                    );
                                }
                                InlineASM::AArch64 => {
                                    if $r * 16 < 450 {
                                        *out += &format!(
                                            "\t\t\"ldp d{}, d{}, {}\\n\\t\"\n",
                                            2 * ($i + 1),
                                            2 * ($i + 1) + 1,
                                            addr_re,
                                        );
                                    } else {
                                        *out += &format!(
                                            "\t\t\"ldr d{}, {}\\n\\t\"\n",
                                            2 * ($i + 1),
                                            addr_re,
                                        );
                                        *out += &format!(
                                            "\t\t\"ldr d{}, {}\\n\\t\"\n",
                                            2 * ($i + 1) + 1,
                                            addr_im,
                                        );
                                    }
                                }
                                InlineASM::None => unreachable!(),
                            }
                        };
                    }

                    macro_rules! mul_complex {
                        ($i: expr, $real: expr) => {
                            match asm_flavour {
                                InlineASM::X64 => {
                                    if $real {
                                        *out += &format!(
                                            "\t\t\"mulpd %%xmm{0}, %%xmm1\\n\\t\"\n",
                                            $i + 1
                                        );
                                    } else {
                                        *out += &format!(
                                            "\t\t\"movapd %%xmm1, %%xmm0\\n\\t\"
\t\t\"unpckhpd %%xmm0, %%xmm0\\n\\t\"
\t\t\"unpcklpd %%xmm1, %%xmm1\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm1\\n\\t\"
\t\t\"shufpd $1, %%xmm0, %%xmm0\\n\\t\"
\t\t\"addsubpd %%xmm0, %%xmm1\\n\\t\"\n",
                                            $i + 1
                                        );
                                    }
                                }
                                InlineASM::AVX2 => {
                                    if $real {
                                        *out += &format!(
                                            "\t\t\"vmulpd %%ymm{0}, %%ymm0\\n\\t\"\n",
                                            $i + 1
                                        );
                                        *out +=
                                            &format!("\t\t\"vxorpd %%ymm1, %%ymm1, %%ymm1\\n\\t\""); // im = 0
                                    } else {
                                        *out += &format!(
                                            "\t\t\"vmulpd %%ymm0, %%ymm{0}, %%ymm14\\n\\t\"
\t\t\"vmulpd %%ymm0, %%ymm{1}, %%ymm15\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm{1}, %%ymm0\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm{0}, %%ymm{1}\\n\\t\"
\t\t\"vsubpd %%ymm0, %%ymm14, %%ymm0\\n\\t\"
\t\t\"vaddpd %%ymm15, %%ymm{1}, %%ymm1\\n\\t\"\n",
                                            2 * $i,
                                            2 * $i + 1,
                                        );
                                    }
                                }
                                InlineASM::AArch64 => {
                                    if $real {
                                        *out += &format!(
                                            "\t\t\"fmul d2, d{}, d2\\n\\t\"\n",
                                            2 * ($i + 1)
                                        );
                                        *out += &format!("\t\t\"fmov d3, xzr\\n\\t\""); // im = 0
                                    } else {
                                        *out += &format!(
                                            "
\t\t\"fmul    d0, d{0}, d3\\n\\t\"
\t\t\"fmul    d1, d{1}, d3\\n\\t\"
\t\t\"fmadd   d3, d{0}, d2, d1\\n\\t\"
\t\t\"fnmsub  d2, d{1}, d2, d0\\n\\t\"\n",
                                            2 * ($i + 1) + 1,
                                            2 * ($i + 1),
                                        )
                                    }
                                }
                                InlineASM::None => unreachable!(),
                            }
                        };
                    }

                    let num_real_args = match c {
                        ComplexPhase::Real => a.len(),
                        ComplexPhase::PartialReal(n) => *n,
                        ComplexPhase::Imag | ComplexPhase::Any => 0,
                    };

                    if !matches!(asm_flavour, InlineASM::AVX2) && a.len() < 15 || a.len() < 8 {
                        for (i, r) in a.iter().enumerate() {
                            load_complex!(i, *r);
                        }

                        for i in 1..a.len() {
                            // optimized complex multiplication
                            mul_complex!(i, i < num_real_args);
                        }
                    } else {
                        load_complex!(0, a[0]);

                        // load multiplications one after the other
                        for (i, r) in a.iter().enumerate().skip(1) {
                            load_complex!(1, *r);
                            mul_complex!(1, i < num_real_args);
                        }
                    }

                    let (addr_re, addr_im) = asm_load!(*o);
                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t\"movupd %%xmm1, {addr_re}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            *out += &format!("\t\t\"vmovupd %%ymm0, {addr_re}\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd %%ymm1, {addr_im}\\n\\t\"\n");
                        }
                        InlineASM::AArch64 => {
                            if *o * 16 < 450 {
                                *out += &format!("\t\t\"stp d2, d3, {addr_re}\\n\\t\"\n");
                            } else {
                                *out += &format!("\t\t\"str d2, {addr_re}\\n\\t\"\n");
                                *out += &format!("\t\t\"str d3, {addr_im}\\n\\t\"\n");
                            }
                        }
                        InlineASM::None => unreachable!(),
                    };
                }
                Instr::Pow(o, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        let addr_b = asm_load!(*b);
                        let addr_o = asm_load!(*o);
                        match asm_flavour {
                            InlineASM::X64 => {
                                if let ComplexPhase::Real = *c {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"movupd {}(%1), %%xmm1\\n\\t\"
\t\t\"divsd %%xmm0, %%xmm1\\n\\t\"
\t\t\"movupd %%xmm1, {}\\n\\t\"\n",
                                        addr_b.0,
                                        (self.reserved_indices - self.param_count + 1) * 16,
                                        addr_o.0
                                    );
                                } else {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"movupd {}(%1), %%xmm1\\n\\t\"
\t\t\"movapd %%xmm0, %%xmm2\\n\\t\"
\t\t\"xorpd %%xmm1, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm2, %%xmm2\\n\\t\"
\t\t\"haddpd %%xmm2, %%xmm2\\n\\t\"
\t\t\"divpd %%xmm2, %%xmm0\\n\\t\"
\t\t\"movupd %%xmm0, {}\\n\\t\"\n",
                                        addr_b.0,
                                        (self.reserved_indices - self.param_count) * 16,
                                        addr_o.0
                                    );
                                }
                            }
                            InlineASM::AVX2 => {
                                if let ComplexPhase::Real = *c {
                                    *out += &format!(
                                        "\t\t\"vmovupd {0}, %%ymm0\\n\\t\"
\t\t\"vmovupd {1}(%1), %%ymm1\\n\\t\"
\t\t\"vdivpd  %%ymm0, %%ymm1, %%ymm0\\n\\t\"
\t\t\"vmovupd %%ymm0, {2}\\n\\t\"
\t\t\"vxorpd %%ymm1, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vmovupd %%ymm1, {3}\\n\\t\"\n",
                                        addr_b.0,
                                        (self.reserved_indices - self.param_count + 1) * 64,
                                        addr_o.0,
                                        addr_o.1
                                    );
                                } else {
                                    // TODO: do FMA on top?
                                    *out += &format!(
                                        "\t\t\"vmovupd {0}, %%ymm0\\n\\t\"
\t\t\"vmovupd {1}, %%ymm1\\n\\t\"
\t\t\"vmulpd %%ymm0, %%ymm0, %%ymm3\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm1, %%ymm4\\n\\t\"
\t\t\"vaddpd %%ymm3, %%ymm4, %%ymm3\\n\\t\"
\t\t\"vdivpd %%ymm3, %%ymm0, %%ymm0\\n\\t\"
\t\t\"vbroadcastsd {2}(%1), %%ymm4\\n\\t\"
\t\t\"vxorpd %%ymm4, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vdivpd %%ymm3, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vmovupd %%ymm0, {3}\\n\\t\"
\t\t\"vmovupd %%ymm1, {4}\\n\\t\"\n",
                                        addr_b.0,
                                        addr_b.1,
                                        (self.reserved_indices - self.param_count) * 64,
                                        addr_o.0,
                                        addr_o.1
                                    );
                                }
                            }
                            InlineASM::AArch64 => {
                                if *b * 16 < 450 {
                                    *out += &format!("\t\t\"ldp d0, d1, {}\\n\\t\"\n", addr_b.0);
                                } else {
                                    *out += &format!("\t\t\"ldr d0, {}\\n\\t\"\n", addr_b.0);
                                    *out += &format!("\t\t\"ldr d1, {}\\n\\t\"\n", addr_b.1);
                                }

                                if let ComplexPhase::Real = *c {
                                    *out += &format!(
                                        "\t\t\"ldr    d2, [%1, {}]\\n\\t\"
\t\t\"fdiv    d0, d2, d0\\n\\t\"\n",
                                        (self.reserved_indices - self.param_count + 1) * 16
                                    );
                                } else {
                                    *out += "
\t\t\"fmul    d2, d0, d0\\n\\t\"
\t\t\"fmadd   d2, d1, d1, d2\\n\\t\"
\t\t\"fneg    d1, d1\\n\\t\"
\t\t\"fdiv    d0, d0, d2\\n\\t\"
\t\t\"fdiv    d1, d1, d2\\n\\t\"\n";
                                }

                                if *o * 16 < 450 {
                                    *out += &format!("\t\t\"stp d0, d1, {}\\n\\t\"\n", addr_o.0);
                                } else {
                                    *out += &format!("\t\t\"str d0, {}\\n\\t\"\n", addr_o.0);
                                    *out += &format!("\t\t\"str d1, {}\\n\\t\"\n", addr_o.1);
                                }
                            }
                            InlineASM::None => unreachable!(),
                        }
                    } else {
                        end_asm_block!(in_asm_block);

                        let base = get_input!(*b);
                        *out += format!("\tZ[{o}] = pow({base}, {e});\n").as_str();
                    }
                }
                Instr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);
                    let base = get_input!(*b);
                    let exp = get_input!(*e);

                    let suffix = if let ComplexPhase::Real = *c {
                        ".real()"
                    } else {
                        ""
                    };

                    *out += format!("\tZ[{o}] = pow({base}{suffix}, {exp}{suffix});\n").as_str();
                }
                Instr::BuiltinFun(o, s, a) => {
                    if in_asm_block
                        && s.get_id() == Symbol::SQRT_ID
                        && let ComplexPhase::Real = *c
                    {
                        let addr_a = asm_load!(*a);
                        let addr_o = asm_load!(*o);

                        match asm_flavour {
                            InlineASM::X64 => {
                                *out += &format!(
                                    "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"sqrtsd %%xmm0, %%xmm0\\n\\t\"
\t\t\"movupd %%xmm0, {}\\n\\t\"\n",
                                    addr_a.0, addr_o.0
                                );
                            }
                            InlineASM::AVX2 => {
                                *out += &format!(
                                    "\t\t\"vmovupd {}, %%ymm0\\n\\t\"
\t\t\"vsqrtpd %%ymm0, %%ymm0\\n\\t\"
\t\t\"vxorpd %%ymm1, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vmovupd %%ymm0, {}\\n\\t\"
\t\t\"vmovupd %%ymm1, {}\\n\\t\"\n",
                                    addr_a.0, addr_o.0, addr_o.1
                                );
                            }
                            InlineASM::AArch64 => {
                                *out += &format!(
                                    "\t\t\"ldr d0, {}\\n\\t\"
\t\t\"fsqrt d0, d0\\n\\t\"
\t\t\"str d0, {}\\n\\t\"\n",
                                    addr_a.0, addr_o.0
                                );
                            }
                            InlineASM::None => unreachable!(),
                        }

                        continue;
                    }

                    end_asm_block!(in_asm_block);

                    let arg = if let ComplexPhase::Real = *c {
                        get_input!(*a) + ".real()"
                    } else {
                        get_input!(*a)
                    };

                    match s.get_id() {
                        Symbol::EXP_ID => {
                            *out += format!("\tZ[{o}] = exp({arg});\n").as_str();
                        }
                        Symbol::LOG_ID => {
                            *out += format!("\tZ[{o}] = log({arg});\n").as_str();
                        }
                        Symbol::SIN_ID => {
                            *out += format!("\tZ[{o}] = sin({arg});\n").as_str();
                        }
                        Symbol::COS_ID => {
                            *out += format!("\tZ[{o}] = cos({arg});\n").as_str();
                        }
                        Symbol::SQRT_ID => {
                            *out += format!("\tZ[{o}] = sqrt({arg});\n").as_str();
                        }
                        Symbol::ABS_ID => {
                            *out += format!("\tZ[{o}] = std::abs({arg});\n").as_str();
                        }
                        Symbol::CONJ_ID => {
                            if let ComplexPhase::Real = *c {
                                *out += format!("\tZ[{o}] = {arg};\n").as_str();
                            } else {
                                *out += format!("\tZ[{o}] = conj({arg});\n").as_str();
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                Instr::ExternalFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let name = &self.external_fns[*s];
                    let args = a.iter().map(|x| get_input!(*x)).collect::<Vec<_>>();

                    *out += format!("\tZ[{}] = {}({});\n", o, name, args.join(", ")).as_str();
                }
                Instr::IfElse(cond, _) => {
                    end_asm_block!(in_asm_block);

                    if asm_flavour == InlineASM::AVX2 {
                        *out += &format!("\tif (all({} != 0.)) {{\n", get_input!(*cond));
                    } else {
                        *out += &format!("\tif ({} != 0.) {{\n", get_input!(*cond));
                    }
                }
                Instr::Goto(_) => {
                    next_label_is_true_branch_end = true;
                }
                Instr::Label(l) => {
                    end_asm_block!(in_asm_block);

                    for (o, b) in label_join_info.get(l).unwrap() {
                        let arg_a = get_input!(*o);
                        let arg_b = get_input!(*b);
                        *out += &format!("\t{} = {};\n", arg_a, arg_b);
                    }

                    if next_label_is_true_branch_end {
                        *out += "\t} else {\n";
                        next_label_is_true_branch_end = false;
                    } else {
                        *out += "\t}\n";
                    }
                }
                Instr::Join(_, _, _, _) => {}
            }
        }

        end_asm_block!(in_asm_block);

        *out += "\t__asm__(\n";
        for (i, r) in &mut self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movupd {}(%2), %%xmm0\\n\\t\"\n", r * 16);
                    }
                    InlineASM::AVX2 => {
                        *out += &format!("\t\t\"vmovupd {}(%2), %%ymm0\\n\\t\"\n", r * 64);
                        *out += &format!("\t\t\"vmovupd {}(%2), %%ymm1\\n\\t\"\n", r * 64 + 32);
                    }
                    InlineASM::AArch64 => {
                        let (addr_re, _) = asm_load!(*r);
                        *out += &format!("\t\t\"ldr q0, {}\\n\\t\"\n", addr_re);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else if *r < self.reserved_indices {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!(
                            "\t\t\"movupd {}(%1), %%xmm0\\n\\t\"\n",
                            (r - self.param_count) * 16
                        );
                    }
                    InlineASM::AVX2 => {
                        *out += &format!(
                            "\t\t\"vmovupd {}(%1), %%ymm0\\n\\t\"\n",
                            (r - self.param_count) * 64
                        );
                        *out += &format!(
                            "\t\t\"vmovupd {}(%1), %%ymm1\\n\\t\"\n",
                            (r - self.param_count) * 64 + 32
                        );
                    }
                    InlineASM::AArch64 => {
                        let (addr_re, _) = asm_load!(*r);
                        *out += &format!("\t\t\"ldr q0, {}\\n\\t\"\n", addr_re);
                    }

                    InlineASM::None => unreachable!(),
                }
            } else {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movupd {}(%0), %%xmm0\\n\\t\"\n", r * 16);
                    }
                    InlineASM::AVX2 => {
                        *out += &format!("\t\t\"vmovupd {}(%0), %%ymm0\\n\\t\"\n", r * 64);
                        *out += &format!("\t\t\"vmovupd {}(%0), %%ymm1\\n\\t\"\n", r * 64 + 32);
                    }
                    InlineASM::AArch64 => {
                        let (addr_re, _) = asm_load!(*r);
                        *out += &format!("\t\t\"ldr q0, {}\\n\\t\"\n", addr_re);
                    }
                    InlineASM::None => unreachable!(),
                }
            }

            match asm_flavour {
                InlineASM::X64 => {
                    *out += &format!("\t\t\"movupd %%xmm0, {}(%3)\\n\\t\"\n", i * 16);
                }
                InlineASM::AVX2 => {
                    *out += &format!("\t\t\"vmovupd %%ymm0, {}(%3)\\n\\t\"\n", i * 64);
                    *out += &format!("\t\t\"vmovupd %%ymm1, {}(%3)\\n\\t\"\n", i * 64 + 32);
                }
                InlineASM::AArch64 => {
                    let dest = i * 16;
                    if dest > 32760 {
                        let d = dest.ilog2();
                        let shift = d.min(12);
                        let coeff = dest / (1 << shift);
                        let rest = dest - (coeff << shift);
                        second_index = 0;
                        *out += &format!("\t\t\"add x8, %3, {}, lsl {}\\n\\t\"\n", coeff, shift);
                        *out += &format!("\t\t\"str q0, [x8, {}]\\n\\t\"\n", rest);
                    } else {
                        *out += &format!("\t\t\"str q0, [%3, {}]\\n\\t\"\n", dest);
                    }
                }
                InlineASM::None => unreachable!(),
            }
        }

        match asm_flavour {
            InlineASM::X64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n"
                );
            }
            InlineASM::AVX2 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n"
                );
            }
            InlineASM::AArch64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n"
                );
            }
            InlineASM::None => unreachable!(),
        }

        in_asm_block
    }
}
