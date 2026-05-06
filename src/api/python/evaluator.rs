use super::*;

/// An optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Evaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonExpressionEvaluator {
    pub rational_constants: Vec<Complex<Rational>>,
    pub eval_complex: ExpressionEvaluator<Complex<f64>>,
    pub eval_real: Option<ExpressionEvaluator<f64>>,
    pub jit_real: Option<JITCompiledEvaluator<f64>>,
    pub jit_complex: Option<JITCompiledEvaluator<Complex<f64>>>,
    pub eval_double_float: Option<ExpressionEvaluator<DoubleFloat>>,
    pub eval_double_float_complex: Option<ExpressionEvaluator<Complex<DoubleFloat>>>,
    pub eval_arb_prec: Option<(u32, ExpressionEvaluator<Float>)>,
    pub eval_arb_prec_complex: Option<(u32, ExpressionEvaluator<Complex<Float>>)>,
    pub jit_compile: bool,
}

impl PythonExpressionEvaluator {
    fn evaluate_double_float<'py>(
        &mut self,
        inputs: Vec<PythonMultiPrecisionFloat>,
    ) -> PyResult<Vec<PythonMultiPrecisionFloat>> {
        if self.rational_constants.iter().any(|c| !c.is_real()) {
            return Err(exceptions::PyValueError::new_err(
                "Evaluator contains complex coefficients. Use evaluate_complex instead.",
            ));
        }

        if self.eval_double_float.is_none() {
            self.eval_double_float = Some(
                self.eval_complex
                    .clone()
                    .set_coeff(&self.rational_constants)
                    .map_coeff(&|x| (&x.re).into()),
            );
        }

        let eval = &mut self.eval_double_float.as_mut().unwrap();

        let inputs = inputs
            .into_iter()
            .map(|x| x.0.to_double_float())
            .collect::<Vec<_>>();
        let mut out = vec![0f64.into(); self.eval_complex.get_output_len()];
        eval.evaluate(&inputs, &mut out);
        Ok(out.into_iter().map(|x| Float::from(x).into()).collect())
    }

    fn evaluate_double_float_complex(
        &mut self,
        inputs: Vec<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>,
    ) -> PyResult<Vec<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>> {
        if self.eval_double_float_complex.is_none() {
            self.eval_double_float_complex = Some(
                self.eval_complex
                    .clone()
                    .set_coeff(&self.rational_constants)
                    .map_coeff(&|x| Complex::new((&x.re).into(), (&x.im).into())),
            );
        }

        let eval = &mut self.eval_double_float_complex.as_mut().unwrap();

        let inputs = inputs
            .into_iter()
            .map(|x| Complex::new(x.0.0.to_double_float(), x.1.0.to_double_float()))
            .collect::<Vec<_>>();
        let mut out =
            vec![Complex::from(DoubleFloat::from(0.)); self.eval_complex.get_output_len()];
        eval.evaluate(&inputs, &mut out);
        Ok(out
            .into_iter()
            .map(|x| (Float::from(x.re).into(), Float::from(x.im).into()))
            .collect())
    }
}

fn reshape_evaluator_inputs<T: Clone>(
    arr: CowArray<'_, T, IxDyn>,
    input_len: usize,
) -> PyResult<CowArray<'_, T, IxDyn>> {
    let arr = if arr.shape().len() == 1 {
        let orig_len = arr.len();

        if input_len == 0 {
            if arr.is_empty() {
                arr.into_shape_with_order((1, 0))
                    .map_err(|_| {
                        exceptions::PyValueError::new_err(format!(
                            "Failed to reshape input array. Expected (_, {}), but got {:?}",
                            input_len,
                            [orig_len],
                        ))
                    })?
                    .into_dyn()
            } else {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Input length mismatch: expected (_, {}), but got {:?}",
                    input_len,
                    [orig_len],
                )));
            }
        } else {
            let target_shape = (orig_len / input_len, input_len);

            let arr = if !arr.is_standard_layout() {
                CowArray::from(arr.as_standard_layout().into_owned())
            } else {
                arr
            };

            arr.into_shape_with_order(target_shape)
                .map_err(|_| {
                    exceptions::PyValueError::new_err(format!(
                        "Failed to reshape input array. Expected (_, {}), but got {:?}",
                        input_len,
                        [orig_len],
                    ))
                })?
                .into_dyn()
        }
    } else {
        arr
    };

    if arr.shape().len() != 2 || arr.shape()[1] != input_len {
        return Err(exceptions::PyValueError::new_err(format!(
            "Input length mismatch: expected (_, {}), but got {:?}",
            input_len,
            arr.shape(),
        )));
    }

    Ok(arr)
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonExpressionEvaluator {
    /// Copy the evaluator.
    pub fn __copy__(&self) -> PythonExpressionEvaluator {
        PythonExpressionEvaluator {
            rational_constants: self.rational_constants.clone(),
            eval_complex: self.eval_complex.clone(),
            eval_real: self.eval_real.clone(),
            jit_real: self.jit_real.clone(),
            jit_complex: self.jit_complex.clone(),
            eval_double_float: self.eval_double_float.clone(),
            eval_double_float_complex: self.eval_double_float_complex.clone(),
            eval_arb_prec: self.eval_arb_prec.clone(),
            eval_arb_prec_complex: self.eval_arb_prec_complex.clone(),
            jit_compile: self.jit_compile.clone(),
        }
    }

    /// Set whether to use JIT compilation.
    fn jit_compile(&mut self, jit_compile: bool) {
        self.jit_compile = jit_compile;
    }

    /// Import an exported evaluator from another thread or machine.
    /// Use `save` to export the evaluator.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, evaluator: Bound<'_, PyBytes>) -> PyResult<Self> {
        let (jit_compile, eval, jit_real, jit_complex): (
            bool,
            ExpressionEvaluator<Complex<Rational>>,
            Option<JITCompiledEvaluator<f64>>,
            Option<JITCompiledEvaluator<Complex<f64>>>,
        ) = bincode::decode_from_slice(evaluator.extract()?, bincode::config::standard())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .0;

        Ok(PythonExpressionEvaluator {
            rational_constants: eval.get_constants().to_vec(),
            eval_complex: eval.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64())),
            eval_real: None,
            jit_real,
            jit_complex,
            eval_double_float: None,
            eval_double_float_complex: None,
            eval_arb_prec: None,
            eval_arb_prec_complex: None,
            jit_compile,
        })
    }

    /// Save the evaluator to a byte string that can be imported in another thread or machine.
    /// The external functions are not exported, so they need to be provided separately when importing.
    /// Use `load` to import the evaluator.
    fn save<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyBytes>> {
        bincode::encode_to_vec(
            &(
                self.jit_compile,
                self.eval_complex
                    .clone()
                    .set_coeff(&self.rational_constants),
                &self.jit_real,
                &self.jit_complex,
            ),
            bincode::config::standard(),
        )
        .map(|a| PyBytes::new(py, &a))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Return the instructions for efficiently evaluating the expression, the length of the list
    /// of temporary variables, and the list of constants. This can be used to generate
    /// code for the expression evaluation in any programming language.
    ///
    /// There are four lists that are used in the evaluation instructions:
    /// - `param`: the list of input parameters.
    /// - `temp`: the list of temporary slots. The size of it is provided as the second return value.
    /// - `const`: the list of constants.
    /// - `out`: the list of outputs.
    ///
    /// The instructions are of the form:
    /// - `('add', ('out', 0), [('const', 1), ('param', 0)], 0)` which means `out[0] = const[1] + param[0]`, where the first `0` arguments are real.
    /// - `('mul', ('out', 0), [('temp', 0), ('param', 0)], 1)` which means `out[0] = temp[0] * param[0]`, where the first `1` arguments are real.
    /// - `('pow', ('out', 0), ('param', 0), -1, true)` which means `out[0] = param[0]^-1` and the output is real (`true`).
    /// - `('powf', ('out', 0), ('param', 0), ('param', 1), false)` which means `out[0] = param[0]^param[1]`.
    /// - `('fun', ('temp', 1), f, ["0"], [('param', 0)], true)` which means `temp[1] = f(0, param[0])` and the output is real (`true`).
    /// - `('assign', ('out', 1), ('const', 2))` which means `out[1] = const[2]`.
    /// - `('if_else', ('temp', 0), 5)` which means `if temp[0] == 0 goto label 5` (false branch).
    /// - `('goto', 10)` which means `goto label 10`.
    /// - `('label', 3)` which means `label 3`.
    /// - `('join', ('out', 0), ('temp', 0), 3, 7)` which means `out[0] = (temp[0] != 0) ? label 3 : label 7`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> (ins, m, c) = E('x^2+5/3+cos(x)').evaluator({}, {}, [S('x')]).get_instructions()
    /// >>>
    /// >>> for x in ins:
    /// >>>     print(x)
    /// >>> print('temp list length:', m)
    /// >>> print('constants:', c)
    ///
    /// yields
    ///
    /// ```log
    /// ('mul', ('out', 0), [('param', 0), ('param', 0), 0])
    /// ('fun', ('temp', 1), cos, ('param', 0), false)
    /// ('add', ('out', 0), [('const', 0), ('out', 0), ('temp', 1)])
    /// temp list length: 2
    /// constants: [5/3]
    /// ```
    fn get_instructions<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<Bound<'py, PyTuple>>, usize, Vec<PythonExpression>)> {
        let (instr, max, _) = self.eval_complex.export_instructions();

        fn slot_to_object(slot: &Slot) -> (&str, usize) {
            match slot {
                Slot::Const(x) => ("const", *x),
                Slot::Param(x) => ("param", *x),
                Slot::Temp(x) => ("temp", *x),
                Slot::Out(x) => ("out", *x),
            }
        }

        let mut v = vec![];
        for i in &instr {
            match i {
                Instruction::Add(o, s, real_args) | Instruction::Mul(o, s, real_args) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            if matches!(i, Instruction::Add(_, _, _)) {
                                "add"
                            } else {
                                "mul"
                            }
                            .into_pyobject(py)?
                            .as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            s.iter()
                                .map(slot_to_object)
                                .collect::<Vec<_>>()
                                .into_pyobject(py)?
                                .as_any(),
                            real_args.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Pow(o, b, e, is_real) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "pow".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(b).into_pyobject(py)?.as_any(),
                            e.into_pyobject(py)?.as_any(),
                            is_real.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Powf(o, b, e, is_real) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "powf".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(b).into_pyobject(py)?.as_any(),
                            slot_to_object(e).into_pyobject(py)?.as_any(),
                            is_real.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Fun(o, b, is_real) => {
                    let (name, tags, s) = &**b;
                    v.push(PyTuple::new(
                        py,
                        [
                            "fun".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            PythonExpression::from(Atom::var(*name))
                                .into_pyobject(py)?
                                .as_any(),
                            tags.into_pyobject(py)?.as_any(),
                            s.iter()
                                .map(slot_to_object)
                                .collect::<Vec<_>>()
                                .into_pyobject(py)?
                                .as_any(),
                            is_real.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Assign(o, r) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "assign".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(r).into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::IfElse(cond, label) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "if_else".into_pyobject(py)?.as_any(),
                            slot_to_object(cond).into_pyobject(py)?.as_any(),
                            label.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Join(o, cond, t, f) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "join".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(cond).into_pyobject(py)?.as_any(),
                            slot_to_object(t).into_pyobject(py)?.as_any(),
                            slot_to_object(f).into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Goto(label) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "goto".into_pyobject(py)?.as_any(),
                            label.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Label(label) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "label".into_pyobject(py)?.as_any(),
                            label.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
            }
        }
        Ok((
            v,
            max,
            self.rational_constants
                .iter()
                .map(|x| Atom::num(x.clone()).into())
                .collect(),
        ))
    }

    /// Merge evaluator `other` into `self`. The parameters must be the same, and
    /// the outputs will be concatenated.
    ///
    /// The optional `cpe_rounds` parameter can be used to limit the number of common
    /// pair elimination rounds after the merge.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e1 = E('x').evaluator({}, {}, [S('x')])
    /// >>> e2 = E('x+1').evaluator({}, {}, [S('x')])
    /// >>> e1.merge(e2)
    /// >>> e1.evaluate([[2.]])
    ///
    /// yields `[2, 3]`.
    #[pyo3(signature = (other, cpe_iterations = None))]
    fn merge(
        &mut self,
        other: PythonExpressionEvaluator,
        cpe_iterations: Option<usize>,
    ) -> PyResult<()> {
        let mut r = self
            .eval_complex
            .clone()
            .set_coeff(&self.rational_constants);

        r.merge(
            other
                .eval_complex
                .clone()
                .set_coeff(&other.rational_constants),
            cpe_iterations,
        )
        .map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not merge evaluators: {e}",))
        })?;

        self.rational_constants = r.get_constants().to_vec();
        self.eval_complex = r.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
        self.eval_real = None;
        self.jit_real = None;
        self.jit_complex = None;
        self.eval_arb_prec = None;
        self.eval_arb_prec_complex = None;

        Ok(())
    }

    /// Evaluate the expression for multiple inputs and return the result.
    /// For best performance, use `numpy` arrays instead of lists.
    ///
    /// On the first call, the expression is compiled to a JIT evaluator for improved performance.
    ///
    /// Examples
    /// --------
    /// Evaluate the function for three sets of inputs:
    ///
    /// >>> from symbolica import *
    /// >>> import numpy as np
    /// >>> ev = E('x * y + 2').evaluator({}, {}, [S('x'), S('y')])
    /// >>> print(ev.evaluate(np.array([1., 2., 3., 4., 5., 6.]).reshape((3, 2))))
    ///
    /// Yields`[[ 4.] [ 8.] [14.]]`
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        if self.rational_constants.iter().any(|c| !c.is_real()) {
            return Err(exceptions::PyValueError::new_err(
                "Evaluator contains complex coefficients. Use evaluate_complex_flat instead.",
            ));
        }

        if self.jit_compile && self.jit_real.is_none()
            || !self.jit_compile && self.eval_real.is_none()
        {
            let real_eval = self.eval_complex.clone().map_coeff(&|x| x.re);

            if self.jit_compile {
                self.jit_real = Some(
                    real_eval
                        .jit_compile()
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                );
            } else {
                self.eval_real = Some(real_eval);
            }
        }

        let arr = reshape_evaluator_inputs(
            CowArray::from(inputs.as_array()),
            self.eval_complex.get_input_len(),
        )?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.eval_complex.get_output_len()][..]);

        if self.jit_compile {
            let eval = self.jit_real.as_mut().unwrap();

            for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
                eval.evaluate(
                    i.as_slice().ok_or_else(|| {
                        exceptions::PyValueError::new_err("Failed to convert input to slice")
                    })?,
                    o.as_slice_mut().unwrap(),
                );
            }
        } else {
            let eval = self.eval_real.as_mut().unwrap();

            for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
                eval.evaluate(
                    i.as_slice().ok_or_else(|| {
                        exceptions::PyValueError::new_err("Failed to convert input to slice")
                    })?,
                    o.as_slice_mut().unwrap(),
                );
            }
        }

        Ok(out.into_pyarray(py))
    }

    /// Evaluate the expression for a single input. The precision of the input parameters is honored, and
    /// all constants are converted to a float with a decimal precision set by `decimal_digit_precision`.
    ///
    /// If `decimal_digit_precision` is set to 32, a much faster evaluation using double-float arithmetic is performed.
    ///
    /// Examples
    /// --------
    /// Evaluate the function for a single input with 50 digits of precision:
    ///
    /// >>> from symbolica import *
    /// >>> ev = E('x^2').evaluator({}, {}, [S('x')])
    /// >>> print(ev.evaluate_with_prec([Decimal('1.234567890121223456789981273238947212312338947923')], 50))
    ///
    /// Yields `1.524157875318369274550121833760353508310334033629`
    #[gen_stub(override_return_type(type_repr = "list[decimal.Decimal]", imports = ("decimal")))]
    fn evaluate_with_prec<'py>(
        &mut self,
        inputs: Vec<PythonMultiPrecisionFloat>,
        decimal_digit_precision: u32,
    ) -> PyResult<Vec<PythonMultiPrecisionFloat>> {
        if decimal_digit_precision == 32 {
            return self.evaluate_double_float(inputs);
        }

        let prec = (decimal_digit_precision as f64 * std::f64::consts::LOG2_10).ceil() as u32;

        if self.rational_constants.iter().any(|c| !c.is_real()) {
            return Err(exceptions::PyValueError::new_err(
                "Evaluator contains complex coefficients. Use evaluate_complex instead.",
            ));
        }

        if self.eval_arb_prec.is_none() || self.eval_arb_prec.as_ref().unwrap().0 != prec {
            self.eval_arb_prec = Some((
                prec,
                self.eval_complex
                    .clone()
                    .set_coeff(&self.rational_constants)
                    .map_coeff(&|x| x.re.to_multi_prec_float(prec)),
            ));
        }

        let eval = &mut self.eval_arb_prec.as_mut().unwrap().1;

        let inputs = inputs.into_iter().map(|x| x.0).collect::<Vec<_>>();
        let mut out = vec![Float::with_val(prec, 0); self.eval_complex.get_output_len()];
        eval.evaluate(&inputs, &mut out);
        Ok(out.into_iter().map(|x| x.into()).collect())
    }

    /// Evaluate the expression for multiple inputs and return the result.
    /// For best performance, use `numpy` arrays and `np.complex128` instead of lists and
    /// `complex`.
    ///
    /// On the first call, the expression is compiled to a JIT evaluator for improved performance.
    ///
    /// Examples
    /// --------
    /// Evaluate the function for three sets of inputs:
    ///
    /// >>> from symbolica import *
    /// >>> import numpy as np
    /// >>> ev = E('x * y + 2').evaluator({}, {}, [S('x'), S('y')])
    /// >>> print(ev.evaluate(np.array([1.+2j, 2., 3., 4., 5., 6.]).reshape((3, 2))))
    ///
    /// Yields`[[ 4.+4.j] [14.+0.j] [32.+0.j]]`
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate_complex<'py>(
        &mut self,
        py: Python<'py>,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, Complex64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        if self.jit_compile && self.jit_complex.is_none() {
            self.jit_complex = Some(
                self.eval_complex
                    .jit_compile()
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            );
        }

        let arr = reshape_evaluator_inputs(
            CowArray::from(inputs.as_array()),
            self.eval_complex.get_input_len(),
        )?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.eval_complex.get_output_len()][..]);

        if self.jit_compile {
            let eval = self.jit_complex.as_mut().unwrap();

            for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
                let sc = unsafe {
                    std::mem::transmute::<&[Complex64], &[Complex<f64>]>(i.as_slice().unwrap())
                };
                let os = unsafe {
                    std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                        o.as_slice_mut().unwrap(),
                    )
                };

                eval.evaluate(sc, os);
            }
        } else {
            for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
                let sc = unsafe {
                    std::mem::transmute::<&[Complex64], &[Complex<f64>]>(i.as_slice().unwrap())
                };
                let os = unsafe {
                    std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                        o.as_slice_mut().unwrap(),
                    )
                };

                self.eval_complex.evaluate(sc, os);
            }
        }
        Ok(out.into_pyarray(py))
    }

    /// Evaluate the expression for a single complex input, represented as a tuple of real and imaginary parts.
    /// The precision of the input parameter is honored, and all constants are converted to a float with a decimal precision set by `decimal_digit_precision`.
    ///
    /// If `decimal_digit_precision` is set to 32, a much faster evaluation using double-float arithmetic is performed.
    ///
    /// Examples
    /// --------
    /// Evaluate the function for a single input with 50 digits of precision:
    ///
    /// >>> from symbolica import *
    /// >>> ev = E('x^2').evaluator({}, {}, [S('x')])
    /// >>> print(ev.evaluate_complex_with_prec(
    /// >>>     [(Decimal('1.234567890121223456789981273238947212312338947923'), Decimal('3.434567890121223356789981273238947212312338947923'))], 50))
    ///
    /// Yields `[(Decimal('-10.27209871653338252296233957800668637617803672307'), Decimal('8.480414467170121512062583245527383392798704790330'))]`
    #[gen_stub(override_return_type(type_repr = "list[tuple[decimal.Decimal, decimal.Decimal]]", imports = ("decimal")))]
    fn evaluate_complex_with_prec<'py>(
        &mut self,
        inputs: Vec<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>,
        decimal_digit_precision: u32,
    ) -> PyResult<Vec<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>> {
        if decimal_digit_precision == 32 {
            return self.evaluate_double_float_complex(inputs);
        }

        let prec = (decimal_digit_precision as f64 * std::f64::consts::LOG2_10).ceil() as u32;

        if self.eval_arb_prec_complex.is_none()
            || self.eval_arb_prec_complex.as_ref().unwrap().0 != prec
        {
            // build a new arb prec evaluator with the desired precision
            self.eval_arb_prec_complex = Some((
                prec,
                self.eval_complex
                    .clone()
                    .set_coeff(&self.rational_constants)
                    .map_coeff(&|x| {
                        Complex::new(
                            x.re.to_multi_prec_float(prec),
                            x.im.to_multi_prec_float(prec),
                        )
                    }),
            ));
        }

        let eval = &mut self.eval_arb_prec_complex.as_mut().unwrap().1;

        let inputs = inputs
            .into_iter()
            .map(|x| Complex::new(x.0.0, x.1.0))
            .collect::<Vec<_>>();
        let mut out = vec![
            Complex::new(Float::with_val(prec, 0), Float::with_val(prec, 0));
            self.eval_complex.get_output_len()
        ];
        eval.evaluate(&inputs, &mut out);
        Ok(out
            .into_iter()
            .map(|x| (x.re.into(), x.im.into()))
            .collect())
    }

    /// Dualize the evaluator to support hyper-dual numbers with the given shape,
    /// indicating the number of derivatives in every variable per term.
    /// This allows for efficient computation of derivatives.
    ///
    /// For example, to compute first derivatives in two variables `x` and `y`,
    /// use `dual_shape = [[0, 0], [1, 0], [0, 1]]`.
    ///
    /// External functions must be mapped to `len(dual_shape)` different functions
    /// that compute a single component each. The input to the functions
    /// is the flattened vector of all components of all parameters,
    /// followed by all previously computed output components.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e1 = E('x^2 + y*x').evaluator({}, {}, [S('x'), S('y')])
    /// >>> e1.dualize([[0, 0], [1, 0], [0, 1]])
    /// >>> r = e1.evaluate([[2., 1., 0., 3., 0., 1.]])
    /// >>> print(r)  # [10, 7, 2]
    ///
    /// Mapping external functions:
    ///
    /// >>> ev = E('f(x + 1)').evaluator({}, {}, [S('x')], external_functions={(S('f'), 'f'): lambda args: args[0]})
    /// >>> ev.dualize([[0], [1]], {('f', 'f0', 0): lambda args: args[0], ('f', 'f1', 1): lambda args: args[1]})
    /// >>> print(ev.evaluate([[2., 1.]]))  # [[3. 1.]]
    ///
    /// Parameters
    /// ----------
    /// dual_shape : list[list[int]]
    ///     The shape of the dual numbers, indicating the number of derivatives
    ///     in every variable per term.
    /// zero_components : Optional[list[tuple[int, int]]]
    ///     A list of components that are known to be zero and can be skipped in the dualization.
    ///     Each component is specified as a tuple of (parameter index, dual index).
    #[pyo3(signature = (dual_shape, zero_components = Vec::new()))]
    fn dualize(
        &mut self,
        dual_shape: Vec<Vec<usize>>,
        zero_components: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        let zero = (0..dual_shape.len())
            .map(|_| Complex::new(Q.zero(), Q.zero()))
            .collect();
        let dual = Dualizer::new(HyperDual::from_values(dual_shape, zero), zero_components);

        let r = self
            .eval_complex
            .clone()
            .set_coeff(&self.rational_constants)
            .vectorize(&dual)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not dualize evaluator: {}", e))
            })?;

        self.rational_constants = r.get_constants().to_vec();
        self.eval_complex = r.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
        self.eval_real = None;
        self.jit_real = None;
        self.jit_complex = None;
        self.eval_arb_prec = None;
        self.eval_arb_prec_complex = None;

        Ok(())
    }

    /// Set which parameters are fully real. This allows for more optimal
    /// assembly output that uses real arithmetic instead of complex arithmetic
    /// where possible.
    ///
    /// You can also set if all encountered sqrt, log, and powf operations with real
    /// arguments are expected to yield real results.
    ///
    /// Must be called after all optimization functions and merging are performed
    /// on the evaluator and before the first call to `evaluate`, or the registration will be lost.
    #[pyo3(signature = (real_params, sqrt_real = false, log_real = false, powf_real = false, verbose = false))]
    fn set_real_params(
        &mut self,
        real_params: Vec<usize>,
        sqrt_real: bool,
        log_real: bool,
        powf_real: bool,
        verbose: bool,
    ) -> PyResult<()> {
        self.eval_complex
            .set_real_params(
                &real_params,
                ComplexEvaluatorSettings::new(sqrt_real, log_real, powf_real, verbose),
            )
            .map_err(|e| exceptions::PyValueError::new_err(e))
    }

    /// Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.
    #[gen_stub(skip)]
    #[pyo3(signature =
        (function_name,
        filename,
        library_name,
        number_type,
        inline_asm = "default",
        optimization_level = 3,
        native = true,
        compiler_path = None,
        compiler_flags = None,
        custom_header = None,
        cuda_number_of_evaluations = 1,
        cuda_block_size = 512
    ))]
    fn compile(
        &self,
        function_name: &str,
        filename: &str,
        library_name: &str,
        number_type: &str,
        inline_asm: &str,
        optimization_level: u8,
        native: bool,
        compiler_path: Option<&str>,
        compiler_flags: Option<Vec<String>>,
        custom_header: Option<String>,
        cuda_number_of_evaluations: usize,
        cuda_block_size: usize,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let mut options = match number_type {
            "real" | "complex" => CompileOptions {
                optimization_level: optimization_level as usize,
                native,
                ..f64::get_default_compile_options()
            },
            "real_4x" | "complex_4x" => CompileOptions {
                optimization_level: optimization_level as usize,
                native,
                ..<wide::f64x4>::get_default_compile_options()
            },
            "cuda_real" | "cuda_complex" => CompileOptions {
                optimization_level: optimization_level as usize,
                ..CudaRealf64::get_default_compile_options()
            },
            _ => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid number type {} specified.",
                    number_type,
                )));
            }
        };

        if let Some(compiler_path) = compiler_path {
            options.compiler = compiler_path.to_string();
        }

        if let Some(compiler_flags) = compiler_flags {
            options.args = compiler_flags;
        }

        let inline_asm = match inline_asm.to_lowercase().as_str() {
            "default" => InlineASM::default(),
            "x64" => InlineASM::X64,
            "avx2" => InlineASM::AVX2,
            "aarch64" => InlineASM::AArch64,
            "none" => InlineASM::None,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Invalid inline assembly type specified.",
                ));
            }
        };

        match number_type {
            "real" => PythonCompiledRealExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<f64>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_complex.get_input_len(),
                output_len: self.eval_complex.get_output_len(),
            }
            .into_py_any(py),
            "complex" => PythonCompiledComplexExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<Complex<f64>>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_complex.get_input_len(),
                output_len: self.eval_complex.get_output_len(),
            }
            .into_py_any(py),
            "real_4x" => PythonCompiledSimdRealExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<wide::f64x4>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_complex.get_input_len(),
                output_len: self.eval_complex.get_output_len(),
            }
            .into_py_any(py),
            "complex_4x" => PythonCompiledSimdComplexExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<Complex<wide::f64x4>>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_complex.get_input_len(),
                output_len: self.eval_complex.get_output_len(),
            }
            .into_py_any(py),
            "cuda_real" => PythonCompiledCudaRealExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<CudaRealf64>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load_with_settings(CudaLoadSettings {
                        number_of_evaluations: cuda_number_of_evaluations,
                        block_size: cuda_block_size,
                    })
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_complex.get_input_len(),
                output_len: self.eval_complex.get_output_len(),
            }
            .into_py_any(py),
            "cuda_complex" => PythonCompiledCudaComplexExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<CudaComplexf64>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load_with_settings(CudaLoadSettings {
                        number_of_evaluations: cuda_number_of_evaluations,
                        block_size: cuda_block_size,
                    })
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_complex.get_input_len(),
                output_len: self.eval_complex.get_output_len(),
            }
            .into_py_any(py),
            _ => Err(exceptions::PyValueError::new_err(format!(
                "Invalid number type {} specified.",
                number_type,
            ))),
        }
    }
}

#[cfg(feature = "python_stubgen")]
static ONE: fn() -> String = || "1".into();
#[cfg(feature = "python_stubgen")]
static THREE: fn() -> String = || "3".into();
#[cfg(feature = "python_stubgen")]
static CUDA_BLOCK_DEFAULT: fn() -> String = || "256".into();
#[cfg(feature = "python_stubgen")]
static DEFAULT: fn() -> String = || "\"default\"".into();

#[cfg(feature = "python_stubgen")]
submit! {
PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpressionEvaluator>,
        attrs: &[],
        getters: &[],
        setters: &[],
        file: "python.rs",
        line: line!(),
        column: column!(),
        methods: &[
            MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['real']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            is_overload: true,
            r#type: MethodType::Class,
            r#return: || PythonCompiledRealExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['complex']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledComplexExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['real_4x']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledSimdRealExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['complex_4x']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledSimdComplexExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['cuda_real']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_number_of_evaluations",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(ONE),
                    type_info: || Option::<usize>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_block_size",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(CUDA_BLOCK_DEFAULT),
                    type_info: || Option::<usize>::type_input(),
                },
            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledCudaRealExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

You may have to specify `-code=sm_XY` for your architecture `XY` in the compiler flags to prevent a potentially long
JIT compilation upon the first evaluation.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code.
cuda_number_of_evaluations: Optional[int]
    The number of parallel evaluations to perform on the CUDA device. The input to evaluate must
    have the length `cuda_number_of_evaluations * arg_len`.
cuda_block_size: Optional[int]
    The block size to use for CUDA kernel launches."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['cuda_complex']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_number_of_evaluations",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(ONE),
                    type_info: || Option::<usize>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_block_size",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(CUDA_BLOCK_DEFAULT),
                    type_info: || Option::<usize>::type_input(),
                },
            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledCudaComplexExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

You may have to specify `-code=sm_XY` for your architecture `XY` in the compiler flags to prevent a potentially long
JIT compilation upon the first evaluation.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code.
cuda_number_of_evaluations: Optional[int]
    The number of parallel evaluations to perform on the CUDA device. The input to evaluate must
    have the length `cuda_number_of_evaluations * arg_len`.
cuda_block_size: Optional[int]
    The block size to use for CUDA kernel launches."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        }

          ],
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "CompiledRealEvaluator",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonCompiledRealExpressionEvaluator {
    pub eval: CompiledRealEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledRealExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledRealEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = reshape_evaluator_inputs(CowArray::from(inputs.as_array()), self.input_len)?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);
        for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            self.eval.evaluate(
                i.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                o.as_slice_mut().unwrap(),
            );
        }

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "CompiledSimdRealEvaluator",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonCompiledSimdRealExpressionEvaluator {
    pub eval: CompiledSimdRealEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledSimdRealExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledSimdRealEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = reshape_evaluator_inputs(CowArray::from(inputs.as_array()), self.input_len)?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        self.eval
            .evaluate_batch(
                n_inputs,
                arr.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                out.as_slice_mut().unwrap(),
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Batch error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "CompiledCudaRealEvaluator",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonCompiledCudaRealExpressionEvaluator {
    pub eval: CompiledCudaRealEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledCudaRealExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[pyo3(signature =
        (filename, function_name, input_len, output_len, number_of_evaluations, block_size = 512))]
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
        number_of_evaluations: usize,
        block_size: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledCudaRealEvaluator::load_with_settings(
                filename,
                function_name,
                CudaLoadSettings {
                    number_of_evaluations,
                    block_size,
                },
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = reshape_evaluator_inputs(CowArray::from(inputs.as_array()), self.input_len)?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        self.eval
            .evaluate(
                arr.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                out.as_slice_mut().unwrap(),
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Evaluation error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "CompiledCudaComplexEvaluator",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonCompiledCudaComplexExpressionEvaluator {
    pub eval: CompiledCudaComplexEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledCudaComplexExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[pyo3(signature =
        (filename, function_name, input_len, output_len, number_of_evaluations, block_size = 512))]
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
        number_of_evaluations: usize,
        block_size: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledCudaComplexEvaluator::load_with_settings(
                filename,
                function_name,
                CudaLoadSettings {
                    number_of_evaluations,
                    block_size,
                },
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, Complex64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = reshape_evaluator_inputs(CowArray::from(inputs.as_array()), self.input_len)?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        let sc = unsafe {
            std::mem::transmute::<&[Complex64], &[Complex<f64>]>(arr.as_slice().ok_or_else(
                || exceptions::PyValueError::new_err("Failed to convert input to slice"),
            )?)
        };
        let os = unsafe {
            std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                out.as_slice_mut().unwrap(),
            )
        };

        self.eval
            .evaluate(sc, os)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Evaluation error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "CompiledComplexEvaluator",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonCompiledComplexExpressionEvaluator {
    pub eval: CompiledComplexEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledComplexExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledComplexEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, Complex64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = reshape_evaluator_inputs(CowArray::from(inputs.as_array()), self.input_len)?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);
        for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            let sc = unsafe {
                std::mem::transmute::<&[Complex64], &[Complex<f64>]>(i.as_slice().unwrap())
            };
            let os = unsafe {
                std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                    o.as_slice_mut().unwrap(),
                )
            };

            self.eval.evaluate(sc, os);
        }

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "CompiledSimdComplexEvaluator",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonCompiledSimdComplexExpressionEvaluator {
    pub eval: CompiledSimdComplexEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledSimdComplexExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledSimdComplexEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLikeDyn<'py, Complex64, AllowTypeChange>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = reshape_evaluator_inputs(CowArray::from(inputs.as_array()), self.input_len)?;

        let n_inputs = arr.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        let sc = unsafe {
            std::mem::transmute::<&[Complex64], &[Complex<f64>]>(arr.as_slice().ok_or_else(
                || exceptions::PyValueError::new_err("Failed to convert input to slice"),
            )?)
        };
        let os = unsafe {
            std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                out.as_slice_mut().unwrap(),
            )
        };

        self.eval
            .evaluate_batch(n_inputs, sc, os)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Batch error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}
