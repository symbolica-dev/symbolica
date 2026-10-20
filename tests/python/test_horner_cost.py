"""Horner scoring options preserve results and final pair elimination."""
from itertools import permutations

import pytest

from symbolica import Expression, S


def _polynomial(a, b, c):
    return (b * c**3 - 3 * b**2 - a * c**2 + 2 * a**2 * c
            - 2 * a**2 * b**2 - 3 * a**3 * c + 2 * a**3 * c**2
            + 3 * a**3 * b**2 * c - 2 * a**3 * b**2 * c**2)


def _build(expression, params, multiple, **settings):
    if multiple:
        return Expression.evaluator_multiple(
            [expression, expression + 1], params, **settings)
    return expression.evaluator(params, **settings)


def _cost(evaluator):
    additions = multiplications = 0
    for instruction in evaluator.get_instructions().instructions:
        if instruction[0] == "add":
            additions += len(instruction[2]) - 1
        elif instruction[0] == "mul":
            multiplications += len(instruction[2]) - 1
        elif instruction[0] == "pow":
            exponent = instruction[3]
            assert exponent > 0
            multiplications += exponent.bit_length() + exponent.bit_count() - 2
        else:
            raise AssertionError(f"Unexpected polynomial instruction: {instruction}")
    return multiplications, additions


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("horner_use_cpe", [False, True])
@pytest.mark.parametrize("direct_translation", [False, True])
def test_horner_cost_option_preserves_numerical_outputs(multiple, horner_use_cpe, direct_translation):
    a, b, c = S("horner_option_a", "horner_option_b", "horner_option_c")
    evaluator = _build(
        _polynomial(a, b, c), [a, b, c], multiple,
        iterations=6, n_cores=1, jit_compile=False,
        direct_translation=direct_translation,
        horner_use_cpe=horner_use_cpe)
    points = [[2., 3., 4.], [-1., 2., -3.], [0.25, -0.5, 0.75]]
    expected = [_polynomial(*point) for point in points]
    rows = [[value, value + 1] if multiple else [value] for value in expected]
    assert evaluator.evaluate(points).tolist() == rows


@pytest.mark.parametrize("multiple", [False, True])
def test_horner_cost_default_and_existing_positional_arguments(multiple):
    a, b, c = S("horner_position_a", "horner_position_b", "horner_position_c")
    expression = _polynomial(a, b, c)
    params = [a, b, c]
    # Every existing optional positional argument retains its original slot.
    arguments = (params, [], 6, None, 1, False, False, True, False, 3,
                 {}, 500, 1000000, 100)
    if multiple:
        constructor = Expression.evaluator_multiple
        arguments = ([expression, expression + 1], *arguments)
    else:
        constructor = expression.evaluator
    default = constructor(*arguments)
    explicit_false = constructor(*arguments, False)
    explicit_true = constructor(*arguments, True)
    assert default.get_instructions().instructions == explicit_false.get_instructions().instructions
    points = [[2., 3., 4.], [0.25, -0.5, 0.75]]
    assert default.evaluate(points).tolist() == explicit_true.evaluate(points).tolist()


@pytest.mark.parametrize("multiple", [False, True])
def test_full_cpe_candidate_cost_can_select_fewer_multiplications(multiple):
    params = list(S("horner_rank_a", "horner_rank_b", "horner_rank_c"))
    improved = False
    # Relabel the same polynomial to avoid depending on symbol interning order.
    for a, b, c in permutations(params):
        expression = _polynomial(a, b, c)
        settings = dict(iterations=6, n_cores=1, jit_compile=False)
        cheap = _build(expression, params, multiple, horner_use_cpe=False, **settings)
        full = _build(expression, params, multiple, horner_use_cpe=True, **settings)
        cheap_cost, full_cost = _cost(cheap), _cost(full)
        # Six candidates exhaust every order of these three variables.
        assert full_cost <= cheap_cost
        improved |= full_cost[0] < cheap_cost[0]
        assert full.evaluate([[2., 3., 4.]]).tolist() == cheap.evaluate([[2., 3., 4.]]).tolist()
    assert improved


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("horner_use_cpe", [False, True])
@pytest.mark.parametrize("direct_translation", [False, True])
def test_final_pair_elimination_is_independent_of_horner_cost(multiple, horner_use_cpe, direct_translation):
    a, b, c, d = S("horner_pair_a", "horner_pair_b", "horner_pair_c", "horner_pair_d")
    expression = (a + b + c) * (a + b + d)
    settings = dict(iterations=1, n_cores=1, jit_compile=False,
                    direct_translation=direct_translation,
                    horner_use_cpe=horner_use_cpe)
    without_pairs = _build(expression, [a, b, c, d], multiple, cpe_iterations=0, **settings)
    with_pairs = _build(expression, [a, b, c, d], multiple, cpe_iterations=None, **settings)
    assert _cost(with_pairs)[1] < _cost(without_pairs)[1]
    assert with_pairs.evaluate([[1., 2., 3., 4.]]).tolist() == (
        [[42., 43.]] if multiple else [[42.]])
