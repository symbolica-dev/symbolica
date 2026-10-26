import pytest

from symbolica import E, S, Evaluator, EvaluatorConstantFunction, Expression, Float


def constant_function_fields(exported):
    return [
        (constant.index, constant.function, constant.tags, constant.arguments)
        for constant in exported.constant_functions
    ]


def test_exported_instructions_list_registered_constants():
    x = S("x")
    exported = E("x + pi").evaluator([x]).get_instructions()
    (constant,) = exported.constant_functions
    assert isinstance(constant, EvaluatorConstantFunction)
    assert constant.function == E("pi")
    assert constant.tags == [] and constant.arguments == []
    assert ("const", constant.index) in exported.instructions[0][2]


def test_exported_instructions_list_fixed_argument_functions():
    x = S("x")
    f = S(
        "constant_functions_square",
        eval={"float": lambda a: a[0] ** 2, "complex": lambda a: a[0] ** 2},
    )
    exported = E("x + constant_functions_square(3)").evaluator([x]).get_instructions()
    (constant,) = exported.constant_functions
    assert constant.function == f
    assert constant.tags == [] and constant.arguments == [E("3")]
    assert ("const", constant.index) in exported.instructions[0][2]


def test_exported_instructions_without_registered_constants():
    x = S("x")
    exported = E("x + 0").evaluator([x]).get_instructions()
    assert exported.constant_functions == []


def test_exported_constant_function_properties_are_read_only():
    exported = E("pi").evaluator([]).get_instructions()
    (constant,) = exported.constant_functions
    before = constant_function_fields(exported)
    for name, value in (
        ("index", 1),
        ("function", E("0")),
        ("tags", [E("1")]),
        ("arguments", [E("2")]),
    ):
        with pytest.raises(AttributeError):
            setattr(constant, name, value)
    assert constant_function_fields(exported) == before
    with pytest.raises(TypeError):
        EvaluatorConstantFunction()


def test_exported_constant_function_preserves_expression_tags():
    x = S("constant_functions_tagged_x")
    label = S("constant_functions_tags::label")
    expected_tags = [E("1/2"), label, label + 2]

    def constant(tags):
        assert tags == expected_tags
        return lambda _arguments, digits: Float("1.25", decimal_digits=digits)

    f = S(
        "constant_functions_tagged_constant",
        eval={"tag_count": len(expected_tags), "constant": constant},
    )
    original = (x + f(*expected_tags)).evaluator([x], jit_compile=False)
    exported = original.get_instructions()
    (constant,) = exported.constant_functions
    assert constant.function == f
    assert all(isinstance(tag, Expression) for tag in constant.tags)
    assert constant.tags == expected_tags
    assert constant.arguments == []
    assert exported.constants[constant.index] == E("0")
    restored = Evaluator.from_instructions(exported)
    value = Float("2.5", decimal_digits=80)
    assert restored.evaluate_with_prec([value], 80) == original.evaluate_with_prec(
        [value], 80
    )
    assert constant_function_fields(restored.get_instructions()) == (
        constant_function_fields(exported)
    )


def test_exported_fixed_argument_function_preserves_expression_tags():
    x = S("constant_functions_tagged_fixed_x")
    label = S("constant_functions_tags::fixed_label")
    expected_tags = [E("2"), label + 1]

    def square(tags):
        assert tags == expected_tags
        return lambda arguments: arguments[0] ** 2

    f = S(
        "constant_functions_tagged_square",
        eval={"tag_count": len(expected_tags), "float": square, "complex": square},
    )
    exported = (x + f(*expected_tags, E("3/2"))).evaluator([x]).get_instructions()
    (constant,) = exported.constant_functions
    assert constant.function == f
    assert all(isinstance(tag, Expression) for tag in constant.tags)
    assert constant.tags == expected_tags
    assert constant.arguments == [E("3/2")]
    assert exported.constants[constant.index] == E("0")

    tags = constant.tags
    arguments = constant.arguments
    tags[0] = E("7")
    arguments.clear()
    assert constant.tags == expected_tags
    assert constant.arguments == [E("3/2")]
    assert constant_function_fields(exported) == [
        (constant.index, f, expected_tags, [E("3/2")])
    ]
    restored = Evaluator.from_instructions(exported)
    assert restored.evaluate([[1.0]]).tolist() == [[3.25]]
