"""Evaluators can be restored from their structured instruction output."""

import math

import numpy as np
import pytest

from symbolica import E, S, Evaluator, Expression, Float, FunctionDefinition


def constant_function_fields(exported):
    return [
        (constant.index, constant.function, constant.tags, constant.arguments)
        for constant in exported.constant_functions
    ]


@pytest.mark.parametrize("jit", [False, True])
def test_instruction_round_trip(jit):
    x = S("instruction_import_x")
    original = Expression.evaluator_multiple(
        [x, E("5/3"), x**2 + E("5/3"), x**2 + E("5/3")],
        [x], jit_compile=False,
    )
    restored = Evaluator.from_instructions(original.get_instructions())
    restored.jit_compile(jit)
    points = [[2.0], [-3.0], [0.25]]
    np.testing.assert_array_equal(restored.evaluate(points), original.evaluate(points))
    complex_points = [[2 + 3j], [-3 + 0.5j]]
    np.testing.assert_array_equal(
        restored.evaluate_complex(complex_points), original.evaluate_complex(complex_points)
    )


def test_instruction_constants_recompute_at_requested_precision():
    x = S("instruction_import_precision_x")
    original = (x + E("pi")).evaluator([x], jit_compile=False)
    restored = Evaluator.from_instructions(original.get_instructions())
    assert restored.evaluate([[1.0]])[0][0] == 1 + math.pi
    for digits in (60, 100):
        value = Float("1.25", decimal_digits=digits)
        assert restored.evaluate_with_prec([value], digits) == original.evaluate_with_prec(
            [value], digits
        )
    # A second export must retain the definitions as well as the placeholders.
    restored_again = Evaluator.from_instructions(restored.get_instructions())
    assert restored_again.evaluate_with_prec([value], digits) == original.evaluate_with_prec(
        [value], digits
    )


def test_instruction_fixed_and_dynamic_external_calls():
    x = S("instruction_import_external_x")
    f = S(
        "instruction_import_external_square",
        eval={"float": lambda args: args[0] ** 2,
              "complex": lambda args: args[0] ** 2},
    )
    original = (f(x) + f(3)).evaluator([x], jit_compile=False)
    restored = Evaluator.from_instructions(original.get_instructions())
    assert restored.evaluate([[2.0], [4.0]]).tolist() == [[13.0], [25.0]]
    assert restored.evaluate_complex([[2 + 1j]]).tolist() == [[12 + 4j]]


def test_instruction_tagged_fixed_and_dynamic_external_calls():
    x = S("instruction_import_tagged_external_x")
    label = S("instruction_import_tags::external_label")
    first_tags = [E("1/2"), label + 1]
    second_tags = [E("1/2"), label + 2]

    def square(tags):
        assert all(isinstance(tag, Expression) for tag in tags)
        assert tags in (first_tags, second_tags)
        offset = 10 if tags == first_tags else 20
        return lambda arguments: arguments[0] ** 2 + offset

    f = S(
        "instruction_import_tagged_external_square",
        eval={"tag_count": 2, "float": square, "complex": square},
    )
    expression = f(*first_tags, x) + f(*second_tags, x) + f(*first_tags, 3)
    original = expression.evaluator([x], jit_compile=False)
    exported = original.get_instructions()
    (constant,) = exported.constant_functions
    assert constant.function == f
    assert constant.tags == first_tags and constant.arguments == [E("3")]
    restored = Evaluator.from_instructions(exported)
    assert restored.evaluate([[2.0], [4.0]]).tolist() == [[57.0], [81.0]]
    assert restored.evaluate_complex([[2 + 1j]]).tolist() == [[55 + 8j]]
    reexported = restored.get_instructions()
    assert constant_function_fields(reexported) == constant_function_fields(exported)
    assert reexported.instructions == exported.instructions
    restored_again = Evaluator.from_instructions(reexported)
    assert restored_again.evaluate([[2.0]]).tolist() == [[57.0]]


@pytest.mark.parametrize("tagged", [False, True])
def test_instruction_retained_function(tagged):
    x, y, f = S("instruction_import_body_x", "instruction_import_body_y",
                "instruction_import_body_f")
    label = S("instruction_import_tags::body_label")
    tags = [E("1/2"), label, label + 2] if tagged else []
    definition = FunctionDefinition(
        f(*tags) if tagged else f, [y], y**2 + E("pi"), inlining="never"
    )
    original = (f(*tags, x) + f(*tags, x + 1)).evaluator(
        [x], functions=[definition], jit_compile=False
    )
    exported = original.get_instructions()
    assert exported.sub_evaluators
    assert exported.sub_evaluators[0].tags == tags
    assert all(
        instruction[3] == tags
        for instruction in exported.instructions
        if instruction[0] == "fun"
    )
    restored = Evaluator.from_instructions(exported)
    assert restored.get_instructions().sub_evaluators[0].tags == tags
    np.testing.assert_array_equal(restored.evaluate([[2.0]]), original.evaluate([[2.0]]))
    assert restored.evaluate_with_prec([Float(2, decimal_digits=80)], 80) == (
        original.evaluate_with_prec([Float(2, decimal_digits=80)], 80)
    )
