"""Run against a freshly built extension with pytest."""

import pytest

from symbolica import E, N, S


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", True),
        ("x", False),
        ("(x+1)^2-x^2-2*x-1", True),
        ("1/x-1/(x+1)-1/(x*(x+1))", True),
        ("sin(x)^2+cos(x)^2-1", True),
        ("log(2)+log(3)-log(6)", None),
        ("x+1/((y+1)^2-y^2-2*y-1)", None),
    ],
)
def test_check_zero_preserves_all_three_results(text, expected):
    assert E(text).check_zero() is expected


def test_check_zero_can_disable_each_stage():
    structural = dict(
        finite_field_samples=0, numerical_samples=0, exact_fallback=False
    )
    assert N(0).check_zero(**structural) is True
    assert S("check_zero_x").check_zero(**structural) is False

    identity = E("(x+1)^2-x^2-2*x-1")
    assert identity.check_zero(**structural) is None
    assert identity.check_zero(
        finite_field_samples=0, numerical_samples=0, exact_fallback=True
    ) is True

    constant = E("sqrt(2)+sqrt(3)")
    assert constant.check_zero(**structural) is None
    assert constant.check_zero(
        finite_field_samples=0, numerical_samples=1, exact_fallback=False
    ) is False

    nonidentity = E("cosh(x)^2-sinh(x)^2")
    assert nonidentity.check_zero(**structural) is None
    assert nonidentity.check_zero(
        finite_field_samples=1, numerical_samples=0, exact_fallback=False
    ) is False


def test_check_zero_precision_is_in_bits_and_reaches_the_requested_limit():
    expression = E("(sqrt(2)+sqrt(3))/10^400")
    numerical_only = dict(
        finite_field_samples=0, numerical_samples=1, exact_fallback=False
    )
    for bits in [0, 127, 128]:
        assert expression.check_zero(max_precision=bits, **numerical_only) is None
    assert expression.check_zero(max_precision=200, **numerical_only) is False


def test_check_zero_can_treat_float_arguments_as_exact_values():
    f = S("check_zero_float_function")
    expression = f(N(0.5)) - f(N(1) / 2)
    assert expression.check_zero() is not True
    assert expression.check_zero(treat_floats_as_exact=True) is True


def test_check_zero_options_are_keyword_only_and_have_unsigned_limits():
    expression = N(0)
    with pytest.raises(TypeError):
        expression.check_zero(1)
    for keyword in ["finite_field_samples", "numerical_samples", "max_precision"]:
        with pytest.raises(OverflowError):
            expression.check_zero(**{keyword: -1})
    with pytest.raises(OverflowError):
        expression.check_zero(max_precision=2**32)
