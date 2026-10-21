"""Check the zero-test signature with ty."""

from typing import assert_type

from symbolica import E

expression = E("(x+1)^2-x^2-2*x-1")
assert_type(expression.check_zero(), bool | None)
assert_type(
    expression.check_zero(
        finite_field_samples=0,
        numerical_samples=1,
        max_precision=200,
        exact_fallback=False,
        treat_floats_as_exact=True,
    ),
    bool | None,
)
