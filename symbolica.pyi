"""
Symbolica is a blazing fast computer algebra system.

It can be used to perform mathematical operations,
such as symbolic differentiation, integration, simplification,
pattern matching and solving equations.

Examples
--------

>>> from symbolica import *
>>> e = E('x^2*log(2*x + y) + exp(3*x)')
>>> a = e.derivative(S('x'))
>>> print("d/dx {} = {}".format(e, a))
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Literal, overload, Iterator, Sequence
from decimal import Decimal
import numpy as np
import numpy.typing as npt


def use_custom_logger() -> None:
    """
    Enable logging using Python's logging module instead of using the default logging.
    This is useful when using Symbolica in a Jupyter notebook or other environments
    where stdout is not easily accessible.

    This function must be called before any Symbolica logging events are emitted.
    """


def get_namespace() -> str:
    """
    Get the Symbolica namespace for the calling module. Use `set_namespace` to set a namespace.
    """


def set_namespace(namespace: str) -> None:
    """
    Set the Symbolica namespace for the calling module.
    All subsequently created symbols in the calling module will be defined within this namespace.

    This function sets the `SYMBOLICA_NAMESPACE` variable in the global scope of the calling module.

    Parameters
    ----------
    namespace: str
        The namespace to set for subsequently created symbols.
    """


def get_version() -> str:
    """
    Get the current Symbolica version.
    """


def is_licensed() -> bool:
    """
    Check if the current Symbolica instance has a valid license key set.
    """


def set_license_key(key: str) -> None:
    """
    Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions
    and before importing any community modules.

    Parameters
    ----------
    key: str
        The license key to register for this machine.
    """


def request_hobbyist_license(name: str, email: str) -> None:
    """
    Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address `email`.

    Parameters
    ----------
    name: str
        The name of the user.
    email: str
        The email address that should receive the license.
    """


def request_trial_license(name: str, email: str, company: str) -> None:
    """
    Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address `email`.

    Parameters
    ----------
    name: str
        The name of the user.
    email: str
        The email address that should receive the license.
    company: str
        The company of the user.
    """


def request_sublicense(name: str, email: str, company: str, super_licence: str) -> None:
    """
    Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
    The key will be sent to the e-mail address `email`.

    Parameters
    ----------
    name: str
        The name of the sublicense user.
    email: str
        The email address that should receive the sublicense.
    company: str
        The company of the sublicense user.
    super_licence: str
        The parent site-wide license key.
    """


def get_license_key(email: str) -> str:
    """
    Get the license key for the account registered with the provided email address.

    Parameters
    ----------
    email: str
        The email address of the licensed account.
    """


@overload
def S(name: str,
      is_symmetric: bool | None = None,
      is_antisymmetric: bool | None = None,
      is_cyclesymmetric: bool | None = None,
      is_linear: bool | None = None,
      is_scalar: bool | None = None,
      is_real: bool | None = None,
      is_integer: bool | None = None,
      is_positive: bool | None = None,
      tags: Sequence[str] | None = None,
      aliases: Sequence[str] | None = None,
      custom_normalization: Transformer | None = None,
      custom_print: Callable[..., str | None] | None = None,
      custom_derivative: Callable[[
          Expression, int], Expression] | None = None,
      data: str | int | Expression | bytes | list[Any] | dict[str | int | Expression, Any] | None = None) -> Expression:
    """
    Create new symbols from `names`. Symbols can have attributes,
    such as symmetries. If no attributes
    are specified and the symbol was previously defined, the attributes are inherited.
    Once attributes are defined on a symbol, they cannot be redefined later.

    Examples
    --------
    Define a regular symbol and use it as a variable:
    >>> x = S('x')
    >>> e = x**2 + 5
    >>> print(e)  # x**2 + 5

    Define a regular symbol and use it as a function:
    >>> f = S('f')
    >>> e = f(1,2)
    >>> print(e)  # f(1,2)


    Define a symmetric function:
    >>> f = S('f', is_symmetric=True)
    >>> e = f(2,1)
    >>> print(e)  # f(1,2)


    Define a linear and symmetric function:
    >>> p1, p2, p3, p4 = ES('p1', 'p2', 'p3', 'p4')
    >>> dot = S('dot', is_symmetric=True, is_linear=True)
    >>> e = dot(p2+2*p3,p1+3*p2-p3)
    dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

    Define a custom normalization function:
    >>> e = S('real_log', custom_normalization=T().replace(E("x_(exp(x1_))"), E("x1_")))
    >>> E("real_log(exp(x)) + real_log(5)")

    Define a custom print function:
    >>> def print_mu(mu: Expression, mode: PrintMode, **kwargs) -> str | None:
    >>>     if mode == PrintMode.Latex:
    >>>         if mu.get_type() == AtomType.Fn:
    >>>             return "\\mu_{" + ",".join(a.format() for a in mu) + "}"
    >>>         else:
    >>>             return "\\mu"
    >>> mu = S("mu", custom_print=print_mu)
    >>> expr = E("mu + mu(1,2)")
    >>> print(expr.to_latex())

    If the function returns `None`, the default print function is used.

    Define a custom derivative function:
    >>> tag = S('tag', custom_derivative=lambda f, index: f)
    >>> x = S('x')
    >>> tag(3, x).derivative(x)

    Add custom data to a symbol:
    >>> x = S('x', data={'my_tag': 'my_value'})
    >>> r = x.get_symbol_data('my_tag')

    Parameters
    ----------
    name : str
        The name of the symbol
    is_symmetric : bool | None
        Set to true if the symbol is symmetric.
    is_antisymmetric : bool | None
        Set to true if the symbol is antisymmetric.
    is_cyclesymmetric : bool | None
        Set to true if the symbol is cyclesymmetric.
    is_linear : bool | None
        Set to true if the symbol is linear.
    is_scalar : bool | None
        Set to true if the symbol is a scalar. It will be moved out of linear functions.
    is_real : bool | None
        Set to true if the symbol is a real number.
    is_integer : bool | None
        Set to true if the symbol is an integer.
    is_positive : bool | None
        Set to true if the symbol is a positive number.
    tags: Sequence[str] | None = None
        A list of tags to associate with the symbol.
    aliases: Sequence[str] | None = None
        A list of aliases to associate with the symbol.
    custom_normalization : Transformer | None
        A transformer that is called after every normalization. Note that the symbol
        name cannot be used in the transformer as this will lead to a definition of the
        symbol. Use a wildcard with the same attributes instead.
    custom_print : Callable[..., str | None] | None:
        A function that is called when printing the variable/function, which is provided as its first argument.
        This function should return a string, or `None` if the default print function should be used.
        The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
    custom_derivative: Callable[[Expression, int], Expression] | None:
        A function that is called when computing the derivative of a function in a given argument.
    data: str | int | Expression | bytes | list | dict | None = None
        Custom user data to associate with the symbol.
    """


@overload
def S(*names: str,
      is_symmetric: bool | None = None,
      is_antisymmetric: bool | None = None,
      is_cyclesymmetric: bool | None = None,
      is_linear: bool | None = None,
      is_scalar: bool | None = None,
      is_real: bool | None = None,
      is_integer: bool | None = None,
      is_positive: bool | None = None,
      tags: Sequence[str] | None = None) -> Sequence[Expression]:
    """
    Create new symbols from `names`. Symbols can have attributes,
    such as symmetries. If no attributes
    are specified and the symbol was previously defined, the attributes are inherited.
    Once attributes are defined on a symbol, they cannot be redefined later.

    Examples
    --------
    Define two regular symbols:
    >>> x, y = S('x', 'y')

    Define two symmetric functions:
    >>> f, g = S('f', 'g', is_symmetric=True)
    >>> e = f(2,1)
    >>> print(e)  # f(1,2)

    Parameters
    ----------
    name : str
        The name of the symbol
    is_symmetric : bool | None
        Set to true if the symbol is symmetric.
    is_antisymmetric : bool | None
        Set to true if the symbol is antisymmetric.
    is_cyclesymmetric : bool | None
        Set to true if the symbol is cyclesymmetric.
    is_linear : bool | None
        Set to true if the symbol is multilinear.
    is_scalar : bool | None
        Set to true if the symbol is a scalar. It will be moved out of linear functions.
    is_real : bool | None
        Set to true if the symbol is a real number.
    is_integer : bool | None
        Set to true if the symbol is an integer.
    is_positive : bool | None
        Set to true if the symbol is a positive number.
    tags: Sequence[str] | None = None
        A list of tags to associate with the symbol.
    """


def N(num: int | float | complex | str | Decimal, relative_error: float | None = None) -> Expression:
    """
    Create a new Symbolica number from an int, a float, or a string.
    A floating point number is kept as a float with the same precision as the input,
    but it can also be converted to the smallest rational number given a `relative_error`.

    Examples
    --------
    >>> e = N(1) / 2
    >>> print(e)  # 1/2

    >>> print(N(1/3))
    >>> print(N(0.33, 0.1))
    >>> print(N('0.333`3'))
    >>> print(N(Decimal('0.1234')))
    3.3333333333333331e-1
    1/3
    3.33e-1
    1.2340e-1

    Parameters
    ----------
    num: int | float | complex | str | Decimal
        The value to convert into a Symbolica number.
    relative_error: float | None
        The maximum relative error used when converting floating-point input to a rational number.
    """


def E(input: str, mode: ParseMode = ParseMode.Symbolica, default_namespace: str | None = None) -> Expression:
    """
    Parse a Symbolica expression from a string.

    Examples
    --------
    >>> e = E('x^2+y+y*4')
    >>> print(e) # x^2+5*y

    Parse a Mathematica expression:
    >>> e = E('Cos[test`x] (2 + 3 I)', mode=ParseMode.Mathematica)
    >>> print(e) # cos(test::x)(2+3i)

    Parameters
    ----------
    input: str
        An input string. UTF-8 characters are allowed.
    mode: ParseMode
        The parsing mode. Use `ParseMode.Mathematica` to parse Mathematica expressions.
    default_namespace: str | None
        The namespace assumed for unqualified symbols during parsing.

    Raises
    ------
    ValueError
        If the input is not a valid expression.
    """


def T() -> Transformer:
    """
    Create a new transformer that maps an expression.
    """


@overload
def P(poly: str, default_namespace: str | None = None, vars: Sequence[Expression] | None = None) -> Polynomial:
    """
    Parse a string a polynomial, optionally, with the variable ordering specified in `vars`.
    All non-polynomial parts will be converted to new, independent variables.

    Parameters
    ----------
    poly: str
        The polynomial expression to parse.
    default_namespace: str | None
        The namespace assumed for unqualified symbols during parsing.
    vars: Sequence[Expression] | None
        The variables to treat as polynomial variables, in the given order.
    """


@overload
def P(poly: str,  minimal_poly: Polynomial, default_namespace: str | None = None, vars: Sequence[Expression] | None = None,
      ) -> NumberFieldPolynomial:
    """
    Parse string to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    All non-polynomial elements will be converted to new independent variables.

    The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
    The minimal polynomial must be a monic, irreducible univariate polynomial.

    Parameters
    ----------
    poly: str
        The polynomial expression to parse.
    minimal_poly: Polynomial
        The minimal polynomial that defines the algebraic extension.
    default_namespace: str | None
        The namespace assumed for unqualified symbols during parsing.
    vars: Sequence[Expression] | None
        The variables to treat as polynomial variables, in the given order.
    """


@overload
def P(poly: str,
      modulus: int,
      default_namespace: str | None = None,
      power: tuple[int, Expression] | None = None,
      minimal_poly: Polynomial | None = None,
      vars: Sequence[Expression] | None = None,
      ) -> FiniteFieldPolynomial:
    """
    Parse a string to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    All non-polynomial elements will be converted to new independent variables.

    The coefficients will be converted to finite field elements modulo `modulus`.
    If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
    `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

    If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.

    Parameters
    ----------
    poly: str
        The polynomial expression to parse.
    modulus: int
        The modulus that defines the finite field.
    default_namespace: str | None
        The namespace assumed for unqualified symbols during parsing.
    power: tuple[int, Expression] | None
        The extension degree and generator that define the finite field.
    minimal_poly: Polynomial | None
        The minimal polynomial that defines the algebraic extension.
    vars: Sequence[Expression] | None
        The variables to treat as polynomial variables, in the given order.
    """


class AtomType(Enum):
    """Specifies the type of the atom."""

    Num = 1
    """The expression is a number."""
    Var = 2
    """The expression is a variable."""
    Fn = 3
    """The expression is a function."""
    Add = 4
    """The expression is a sum."""
    Mul = 5
    """The expression is a product."""
    Pow = 6
    """The expression is a power."""


class SymbolAttribute(Enum):
    """Specifies the attributes of a symbol."""

    Symmetric = 1,
    """ The function is symmetric. """
    Antisymmetric = 2,
    """ The function is antisymmetric."""
    Cyclesymmetric = 3,
    """ The function is cyclesymmetric."""
    Linear = 4,
    """The function is linear."""
    Scalar = 5,
    """The symbol represents a scalar. It will be moved out of linear functions."""
    Real = 6,
    """The symbol represents a real number."""
    Integer = 7,
    """The symbol represents an integer."""
    Positive = 8,
    """The symbol represents a positive number."""


class AtomTree:
    """
    A Python representation of a Symbolica expression.
    The type of the atom is provided in `atom_type`.

    The `head` contains the string representation of:
    - a number if the type is `Num`
    - the variable if the type is `Var`
    - the function name if the type is `Fn`
    - otherwise it is `None`.

    The tail contains the child atoms:
    - the summand for type `Add`
    - the factors for type `Mul`
    - the base and exponent for type `Pow`
    - the function arguments for type `Fn`
    """

    atom_type: AtomType
    """ The type of this atom."""
    head: str | None
    """The string data of this atom."""
    tail: list[AtomTree]
    """The list of child atoms of this atom."""


class ParseMode(Enum):
    """Specifies the parse mode."""

    Symbolica = 1
    """Parse using Symbolica notation."""
    Mathematica = 2
    """Parse using Mathematica notation."""


class PrintMode(Enum):
    """Specifies the print mode."""

    Symbolica = 1
    """Print using Symbolica notation."""
    Latex = 2
    """Print using LaTeX notation."""
    Mathematica = 3
    """Print using Mathematica notation."""
    Sympy = 4
    """Print using Sympy notation."""
    Typst = 5
    """Print using Typst notation."""


class Expression:
    """
    A Symbolica expression.

    Supports standard arithmetic operations, such
    as addition and multiplication.

    Examples
    --------
    >>> x = S('x')
    >>> e = x**2 + 2 - x + 1 / x**4
    >>> print(e)
    """

    E: Expression
    """Euler's number `e`."""

    PI: Expression
    """The mathematical constant `π`."""

    I: Expression
    """The mathematical constant `i`, where `i^2 = -1`."""

    INFINITY: Expression
    """The number that represents infinity: `∞`."""

    COMPLEX_INFINITY: Expression
    """The number that represents infinity with an unknown complex phase: `∞`."""

    INDETERMINATE: Expression
    """The number that represents indeterminacy: `¿`."""

    COEFF: Expression
    """The built-in function that convert a rational polynomials to a coefficient."""

    COS: Expression
    """The built-in cosine function."""

    SIN: Expression
    """The built-in sine function."""

    EXP: Expression
    """The built-in exponential function."""

    LOG: Expression
    """The built-in logarithm function."""

    SQRT: Expression
    """The built-in square root function."""

    ABS: Expression
    """The built-in absolute value function."""

    CONJ: Expression
    """The built-in complex conjugate function."""

    IF: Expression
    """The built-in function for piecewise-defined expressions. `IF(cond, true_expr, false_expr)` evaluates to `true_expr` if `cond` is non-zero and `false_expr` otherwise."""

    @overload
    @classmethod
    def symbol(_cls,
               name: str,
               is_symmetric: bool | None = None,
               is_antisymmetric: bool | None = None,
               is_cyclesymmetric: bool | None = None,
               is_linear: bool | None = None,
               is_scalar: bool | None = None,
               is_real: bool | None = None,
               is_integer: bool | None = None,
               is_positive: bool | None = None,
               tags: Sequence[str] | None = None,
               aliases: Sequence[str] | None = None,
               custom_normalization: Transformer | None = None,
               custom_print: Callable[..., str | None] | None = None,
               custom_derivative: Callable[[
                   Expression, int], Expression] | None = None,
               data: str | int | Expression | bytes | list[Any] | dict[str | int | Expression, Any] | None = None) -> Expression:
        """
        Create new symbols from `names`. Symbols can have attributes,
        such as symmetries. If no attributes
        are specified and the symbol was previously defined, the attributes are inherited.
        Once attributes are defined on a symbol, they cannot be redefined later.

        Examples
        --------
        Define a regular symbol and use it as a variable:
        >>> x = S('x')
        >>> e = x**2 + 5
        >>> print(e)  # x**2 + 5

        Define a regular symbol and use it as a function:
        >>> f = S('f')
        >>> e = f(1,2)
        >>> print(e)  # f(1,2)


        Define a symmetric function:
        >>> f = S('f', is_symmetric=True)
        >>> e = f(2,1)
        >>> print(e)  # f(1,2)


        Define a linear and symmetric function:
        >>> p1, p2, p3, p4 = S('p1', 'p2', 'p3', 'p4')
        >>> dot = S('dot', is_symmetric=True, is_linear=True)
        >>> e = dot(p2+2*p3,p1+3*p2-p3)
        dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

        Define a custom normalization function:
        >>> e = S('real_log', custom_normalization=T().replace(E("x_(exp(x1_))"), E("x1_")))
        >>> E("real_log(exp(x)) + real_log(5)")

        Define a custom print function:
        >>> def print_mu(mu: Expression, latex: bool, **kwargs) -> str | None:
        >>>     if latex:
        >>>         if mu.get_type() == AtomType.Fn:
        >>>             return "\\mu_{" + ",".join(a.format() for a in mu) + "}"
        >>>         else:
        >>>             return "\\mu"
        >>> mu = S("mu", custom_print=print_mu)
        >>> expr = E("mu + mu(1,2)")
        >>> print(expr.to_latex())

        If the function returns `None`, the default print function is used.

        Define a custom derivative function:
        >>> tag = S('tag', custom_derivative=lambda f, index: f)
        >>> x = S('x')
        >>> tag(3, x).derivative(x)

        Add custom data to a symbol:
        >>> x = S('x', data={'my_tag': 'my_value'})
        >>> r = x.get_symbol_data('my_tag')

        Parameters
        ----------
        name : str
            The name of the symbol
        is_symmetric : bool | None
            Set to true if the symbol is symmetric.
        is_antisymmetric : bool | None
            Set to true if the symbol is antisymmetric.
        is_cyclesymmetric : bool | None
            Set to true if the symbol is cyclesymmetric.
        is_linear : bool | None
            Set to true if the symbol is linear.
        is_scalar : bool | None
            Set to true if the symbol is a scalar. It will be moved out of linear functions.
        is_real : bool | None
            Set to true if the symbol is a real number.
        is_integer : bool | None
            Set to true if the symbol is an integer.
        is_positive : bool | None
            Set to true if the symbol is a positive number.
        tags: Sequence[str] | None
            A list of tags to associate with the symbol.
        aliases: Sequence[str] | None
            A list of aliases to associate with the symbol.
        custom_normalization : Transformer | None
            A transformer that is called after every normalization. Note that the symbol
            name cannot be used in the transformer as this will lead to a definition of the
            symbol. Use a wildcard with the same attributes instead.
        custom_print : Callable[..., str | None] | None:
            A function that is called when printing the variable/function, which is provided as its first argument.
            This function should return a string, or `None` if the default print function should be used.
            The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
        custom_derivative: Callable[[Expression, int], Expression] | None:
            A function that is called when computing the derivative of a function in a given argument.
        data: str | int | Expression | bytes | list | dict | None = None
            Custom user data to associate with the symbol.
        """

    @overload
    @classmethod
    def symbol(_cls,
               *names: str,
               is_symmetric: bool | None = None,
               is_antisymmetric: bool | None = None,
               is_cyclesymmetric: bool | None = None,
               is_linear: bool | None = None,
               is_real: bool | None = None,
               is_scalar: bool | None = None,
               is_integer: bool | None = None,
               is_positive: bool | None = None,
               tags: Sequence[str] | None = None) -> Sequence[Expression]:
        """
        Create new symbols from `names`. Symbols can have attributes,
        such as symmetries. If no attributes
        are specified and the symbol was previously defined, the attributes are inherited.
        Once attributes are defined on a symbol, they cannot be redefined later.

        Examples
        --------
        Define two regular symbols:
        >>> x, y = S('x', 'y')

        Define two symmetric functions:
        >>> f, g = S('f', 'g', is_symmetric=True)
        >>> e = f(2,1)
        >>> print(e)  # f(1,2)

        Parameters
        ----------
        name : str
            The name of the symbol
        is_symmetric : bool | None
            Set to true if the symbol is symmetric.
        is_antisymmetric : bool | None
            Set to true if the symbol is antisymmetric.
        is_cyclesymmetric : bool | None
            Set to true if the symbol is cyclesymmetric.
        is_linear : bool | None
            Set to true if the symbol is multilinear.
        is_scalar : bool | None
            Set to true if the symbol is a scalar. It will be moved out of linear functions.
        is_real : bool | None
            Set to true if the symbol is a real number.
        is_integer : bool | None
            Set to true if the symbol is an integer.
        is_positive : bool | None
            Set to true if the symbol is a positive number.
        tags: Sequence[str] | None
            A list of tags to associate with the symbol.
        """

    @overload
    def __call__(self, *args: Expression | int | float | complex | Decimal) -> Expression:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x, f = S('x', 'f')
        >>> e = f(3,x)
        >>> print(e)  # f(3,x)

        Parameters
        ----------
        args: Expression | int | float | complex | Decimal
            The arguments passed to the expression call.
        """

    @overload
    def __call__(self, *args: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x, f = S('x', 'f')
        >>> e = f(3,x)
        >>> print(e)  # f(3,x)

        Parameters
        ----------
        args: HeldExpression | Expression | int | float | complex | Decimal
            The arguments passed to the expression or transformer call.
        """

    @classmethod
    def num(_cls, num: int | float | complex | str | Decimal, relative_error: float | None = None) -> Expression:
        """
        Create a new Symbolica number from an int, a float, or a string.
        A floating point number is kept as a float with the same precision as the input,
        but it can also be converted to the smallest rational number given a `relative_error`.

        Examples
        --------
        >>> e = Expression.num(1) / 2
        >>> print(e)  # 1/2

        >>> print(Expression.num(1/3))
        >>> print(Expression.num(0.33, 0.1))
        >>> print(Expression.num('0.333`3'))
        >>> print(Expression.num(Decimal('0.1234')))
        3.3333333333333331e-1
        1/3
        3.33e-1
        1.2340e-1

        Parameters
        ----------
        num: int | float | complex | str | Decimal
            The value to convert into a Symbolica number.
        relative_error: float | None
            The maximum relative error used when converting floating-point input to a rational number.
        """

    @classmethod
    def get_all_symbol_names(_cls) -> list[str]:
        """
        Return all defined symbol names (function names and variables).
        """

    @classmethod
    def parse(_cls, input: str, mode: ParseMode = ParseMode.Symbolica, default_namespace: str | None = None) -> Expression:
        """
        Parse a Symbolica expression from a string.

        Examples
        --------
        >>> e = E('x^2+y+y*4')
        >>> print(e) # x^2+5*y

        Parse a Mathematica expression:
        >>> e = E('Cos[test`x] (2 + 3 I)', mode=ParseMode.Mathematica)
        >>> print(e) # cos(test::x)(2+3i)

        Parameters
        ----------
        input: str
            An input string. UTF-8 characters are allowed.
        mode: ParseMode
            The parsing mode. Use `ParseMode.Mathematica` to parse Mathematica expressions.
        default_namespace: str
            The namespace assumed for unqualified symbols during parsing.

        Raises
        ------
        ValueError
            If the input is not a valid expression.
        """

    def __new__(cls) -> Expression:
        """
        Create a new expression that represents 0.
        """

    def __copy__(self) -> Expression:
        """
        Copy the expression.
        """

    def __str__(self) -> str:
        """
        Convert the expression into a human-readable string.
        """

    def to_canonical_string(self) -> str:
        """
        Convert the expression into a canonical string that
        is independent on the order of the variables and other
        implementation details.
        """

    def to_int(self) -> int:
        """
        Convert the expression to an integer if possible.
        Raises a `ValueError` if the expression is not an integer.

        Examples
        --------
        >>> e = E('7')
        >>> n = e.to_int()
        """

    @classmethod
    def load(_cls, filename: str, conflict_fn: Callable[[str], str] | None = None) -> Expression:
        """
        Load an expression and its state from a file. The state will be merged
        with the current one. If a symbol has conflicting attributes, the conflict
        can be resolved using the renaming function `conflict_fn`.

        Expressions can be saved using `Expression.save`.

        Examples
        --------
        If `export.dat` contains a serialized expression: `f(x)+f(y)`:
        >>> e = Expression.load('export.dat')

        whill yield `f(x)+f(y)`.

        If we have defined symbols in a different order:
        >>> y, x = S('y', 'x')
        >>> e = Expression.load('export.dat')

        we get `f(y)+f(x)`.

        If we define a symbol with conflicting attributes, we can resolve the conflict
        using a renaming function:

        >>> x = S('x', is_symmetric=True)
        >>> e = Expression.load('export.dat', lambda x: x + '_new')
        print(e)

        will yield `f(x_new)+f(y)`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        conflict_fn: Callable[[str], str] | None
            A callback that resolves symbol conflicts during loading.
        """

    def save(self, filename: str, compression_level: int = 9):
        """
        Save the expression and its state to a binary file.
        The data is compressed and the compression level can be set between 0 and 11.

        The data can be loaded using `Expression.load`.

        Examples
        --------
        >>> e = E("f(x)+f(y)").expand()
        >>> e.save('export.dat')

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        compression_level: int
            The compression level for serialized output.
        """

    def get_byte_size(self) -> int:
        """
        Get the number of bytes that this expression takes up in memory.
        """

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        bracket_level_colors: Sequence[int] | None = [
            244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = 100,
        custom_print_mode: int | None = None,
    ) -> str:
        """
        Convert the expression into a human-readable string, with tunable settings.

        Examples
        --------
        >>> a = E('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
        >>> print(a.format(number_thousands_separator='_', multiplication_operator=' '))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.

        >>> print(E('x^2 + f(x)').format(PrintMode.Sympy))

        yields `x**2+f(x)`

        >>> print(E('x^2 + f(x)').format(PrintMode.Mathematica))

        yields `x^2 + f[x]`

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        terms_on_new_line: bool
            Whether wrapped output should place terms on separate lines.
        color_top_level_sum: bool
            Whether top-level sums should be colorized.
        color_builtin_symbols: bool
            Whether built-in symbols should be colorized.
        bracket_level_colors: Sequence[int] | None
            The colors assigned to successive nested bracket levels.
        print_ring: bool
            Whether the coefficient ring should be included in the printed output.
        symmetric_representation_for_finite_field: bool
            Whether finite-field elements should be printed using symmetric representatives.
        explicit_rational_polynomial: bool
            Whether rational polynomials should be printed explicitly as numerator and denominator.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def format_plain(self) -> str:
        """
        Convert the expression into a plain string, useful for importing and exporting.

        Examples
        --------
        >>> a = E('5 + x^2')
        >>> print(a.format_plain())

        Yields `5 + x^2`, without any coloring.
        """

    def to_latex(self) -> str:
        """
        Convert the expression into a LaTeX string.

        Examples
        --------
        >>> a = E('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
        >>> print(a.to_latex())

        Yields `$$z^{34}+x^{x+2}+y^{4}+f(x,x^{2})+128378127123 z^{\\frac{2}{3}} w^{2} \\frac{1}{x} \\frac{1}{y}+\\frac{3}{5}$$`.
        """

    def to_typst(self, show_namespaces: bool = False) -> str:
        """
        Convert the expression into a Typst string.

        Examples
        --------
        >>> a = E('f(x+2i + 3) * 2 / x')
        >>> print(a.to_typst())

        Yields ```(2 op("f")(3+2𝑖+"x"))/"x"```.

        Parameters
        ----------
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        """

    def to_sympy(self) -> str:
        """
        Convert the expression into a sympy-parsable string.

        Examples
        --------
        >>> from sympy import *
        >>> s = sympy.parse_expr(E('x^2+f((1+x)^y)').to_sympy())
        """

    def to_mathematica(self, show_namespaces: bool = True) -> str:
        """
        Convert the expression into a Mathematica-parsable string.

        Examples
        --------
        >>> a = E('cos(x+2i + 3)+sqrt(conj(x)) + test::y')
        >>> print(a.to_mathematica(show_namespaces=True))

        Yields ```test`y+Cos[x+3+2I]+Sqrt[Conjugate[x]]```.

        Parameters
        ----------
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        """

    def __hash__(self) -> int:
        """
        Hash the expression.
        """

    def get_type(self) -> AtomType:
        """
        Get the type of the atom.
        """

    def to_atom_tree(self) -> AtomTree:
        """
        Convert the expression to a tree.
        """

    def get_name(self) -> str:
        """
        Get the name of a variable or function if the current atom
        is a variable or function, otherwise throw an error.
        """

    def get_tags(self) -> list[str]:
        """
        Get the tags of a variable or function if the current atom
        is a variable or function, otherwise throw an error.
        """

    def get_attributes(self) -> list[SymbolAttribute]:
        """
        Get the attributes of a variable or function if the current atom
        is a variable or function, otherwise throw an error.
        """

    def get_symbol_data(self, key: str | int | Expression | None = None) -> str | int | Expression | bytes | dict[str | int | Expression, Any] | list[Any]:
        """
        Get the data of a variable or function if the current atom
        is a variable or function, otherwise throw an error.
        Optionally, provide a key to access a specific entry in the data map, if
        the data is a map.

        Examples
        --------
        >>> x = S('x', data={'my_tag': 'my_value'})
        >>> print(x.get_symbol_data('my_tag'))  # my_value
        >>> y = S('y', data=3)
        >>> print(y.get_symbol_data())  # 3
        Parameters
        ----------
        key: str | int | Expression | None
            The symbol-data key to retrieve. Omit it to return all stored data.
        """

    def is_scalar(self) -> bool:
        """
        Check if the expression is a scalar. Symbols must have the scalar attribute.

        Examples
        --------
        >>> x = S('x', is_scalar=True)
        >>> e = (x + 1)**2 + 5
        >>> print(e.is_scalar())  # True
        """

    def is_real(self) -> bool:
        """
        Check if the expression is real. Symbols must have the real attribute.

        Examples
        --------
        >>> x = S('x', is_real=True)
        >>> e = (x + 1)**2 / 2 + 5
        >>> print(e.is_real())  # True
        """

    def is_integer(self) -> bool:
        """
        Check if the expression is integer. Symbols must have the integer attribute.

        Examples
        --------
        >>> x = S('x', is_integer=True)
        >>> e = (x + 1)**2 + 5
        >>> print(e.is_integer())  # True
        """

    def is_positive(self) -> bool:
        """
        Check if the expression is a positive scalar. Symbols must have the positive attribute.

        Examples
        --------
        >>> x = S('x', is_positive=True)
        >>> e = (x + 1)**2 + 5
        >>> print(e.is_positive())  # True
        """

    def is_finite(self) -> bool:
        """
        Check if the expression has no infinities and is not indeterminate.

        Examples
        --------
        >>> e = E('1/x + x^2 + log(0)')
        >>> print(e.is_finite())  # False
        """

    def is_constant(self) -> bool:
        """
        Check if the expression is constant, i.e. contains no user-defined symbols or functions.

        Examples
        --------
        >>> e = E('cos(2 + exp(3)) + 5')
        >>> print(e.is_constant())  # True
        """

    def __add__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Add this expression to `other`, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __radd__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Add this expression to `other`, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __sub__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Subtract `other` from this expression, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __rsub__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Subtract this expression from `other`, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __mul__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Multiply this expression with `other`, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __rmul__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Multiply this expression with `other`, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __truediv__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Divide this expression by `other`, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __rtruediv__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Divide `other` by this expression, returning the result.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __pow__(self, exp: Expression | int | float | complex | Decimal) -> Expression:
        """
        Take `self` to power `exp`, returning the result.

        Parameters
        ----------
        exp: Expression | int | float | complex | Decimal
            The exponent.
        """

    def __rpow__(self, base: Expression | int | float | complex | Decimal) -> Expression:
        """
        Take `base` to power `self`, returning the result.

        Parameters
        ----------
        base: Expression | int | float | complex | Decimal
            The base expression.
        """

    def __xor__(self, a: Any) -> Expression:
        """
        Returns a warning that `**` should be used instead of ` ^ ` for taking a power.

        Parameters
        ----------
        a: Any
            The operand passed with `^`; use `**` for exponentiation instead.
        """

    def __rxor__(self, a: Any) -> Expression:
        """
        Returns a warning that `**` should be used instead of ` ^ ` for taking a power.

        Parameters
        ----------
        a: Any
            The operand passed with `^`; use `**` for exponentiation instead.
        """

    def __neg__(self) -> Expression:
        """
        Negate the current expression, returning the result.
        """

    def __len__(self) -> int:
        """
        Return the number of terms in this expression.
        """

    def cos(self) -> Expression:
        """
        Take the cosine of this expression, returning the result.
        """

    def sin(self) -> Expression:
        """
        Take the sine of this expression, returning the result.
        """

    def tan(self) -> Expression:
        """
        Take the tangent of this expression, returning the result.
        """

    def cot(self) -> Expression:
        """
        Take the cotangent of this expression, returning the result.
        """

    def sec(self) -> Expression:
        """
        Take the secant of this expression, returning the result.
        """

    def csc(self) -> Expression:
        """
        Take the cosecant of this expression, returning the result.
        """

    def asin(self) -> Expression:
        """
        Take the inverse sine of this expression, returning the result.
        """

    def acos(self) -> Expression:
        """
        Take the inverse cosine of this expression, returning the result.
        """

    def atan(self) -> Expression:
        """
        Take the inverse tangent of this expression, returning the result.
        """

    def acot(self) -> Expression:
        """
        Take the inverse cotangent of this expression, returning the result.
        """

    def asec(self) -> Expression:
        """
        Take the inverse secant of this expression, returning the result.
        """

    def acsc(self) -> Expression:
        """
        Take the inverse cosecant of this expression, returning the result.
        """

    def sinh(self) -> Expression:
        """
        Take the hyperbolic sine of this expression, returning the result.
        """

    def cosh(self) -> Expression:
        """
        Take the hyperbolic cosine of this expression, returning the result.
        """

    def tanh(self) -> Expression:
        """
        Take the hyperbolic tangent of this expression, returning the result.
        """

    def coth(self) -> Expression:
        """
        Take the hyperbolic cotangent of this expression, returning the result.
        """

    def sech(self) -> Expression:
        """
        Take the hyperbolic secant of this expression, returning the result.
        """

    def csch(self) -> Expression:
        """
        Take the hyperbolic cosecant of this expression, returning the result.
        """

    def asinh(self) -> Expression:
        """
        Take the inverse hyperbolic sine of this expression, returning the result.
        """

    def acosh(self) -> Expression:
        """
        Take the inverse hyperbolic cosine of this expression, returning the result.
        """

    def atanh(self) -> Expression:
        """
        Take the inverse hyperbolic tangent of this expression, returning the result.
        """

    def acoth(self) -> Expression:
        """
        Take the inverse hyperbolic cotangent of this expression, returning the result.
        """

    def asech(self) -> Expression:
        """
        Take the inverse hyperbolic secant of this expression, returning the result.
        """

    def acsch(self) -> Expression:
        """
        Take the inverse hyperbolic cosecant of this expression, returning the result.
        """

    def exp(self) -> Expression:
        """
        Take the exponential of this expression, returning the result.
        """

    def log(self) -> Expression:
        """
        Take the logarithm of this expression, returning the result.
        """

    def sqrt(self) -> Expression:
        """
        Take the square root of this expression, returning the result.
        """

    def abs(self) -> Expression:
        """
        Take the absolute value of this expression, returning the result.
        """

    def gamma(self) -> Expression:
        """
        Apply the gamma function to this expression.
        """

    def digamma(self) -> Expression:
        """
        Apply the digamma function to this expression.
        """

    def polygamma(self, n: Expression | int | float | Decimal) -> Expression:
        """
        Apply the polygamma function of order `n` to this expression.
        """

    def polylog(self, s: Expression | int | float | Decimal) -> Expression:
        """
        Apply the polylogarithm of order `s` to this expression.
        """

    def bessel_j(self, nu: Expression | int | float | Decimal) -> Expression:
        """
        Apply the cylindrical Bessel function of the first kind of order `nu` to this expression.
        """

    def bessel_y(self, nu: Expression | int | float | Decimal) -> Expression:
        """
        Apply the cylindrical Bessel function of the second kind of order `nu` to this expression.
        """

    def bessel_i(self, nu: Expression | int | float | Decimal) -> Expression:
        """
        Apply the modified Bessel function of the first kind of order `nu` to this expression.
        """

    def bessel_k(self, nu: Expression | int | float | Decimal) -> Expression:
        """
        Apply the modified Bessel function of the second kind of order `nu` to this expression.
        """

    def conj(self) -> Expression:
        """
        Take the complex conjugate of this expression, returning the result.

        Examples
        --------
        >>> e = E('x+2 + 3^x + (5+2i) * (test::{real}::real) + (-2)^x')
        >>> print(e.conj())

        Yields `(5-2𝑖)*real+3^conj(x)+conj(x)+conj((-2)^x)+2`.
        """

    def hold(self, t: Transformer) -> HeldExpression:
        """
        Create a held expression that delays the execution of the transformer `t` until the
        resulting held expression is called. Held expressions can be composed like regular expressions
        and are useful for the right-hand side of pattern matching, to act a transformer
        on a wildcard *after* it has been substituted.

        Examples
        -------
        >>> f, x, x_ = S('f', 'x', 'x_')
        >>> e = f((x+1)**2)
        >>> e = e.replace(f(x_), f(x_.hold(T().expand())))

        Parameters
        ----------
        t: Transformer
            The transformer to bind to the expression.
        """

    def contains(self, a: Transformer | HeldExpression | Expression | int | float | Decimal) -> Condition:
        """
        Returns true iff `self` contains `a` literally.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, z = S('x', 'y', 'z')
        >>> e = x * y * z
        >>> e.contains(x) # True
        >>> e.contains(x*y*z) # True
        >>> e.contains(x*y) # False

        Parameters
        ----------
        a: Transformer | HeldExpression | Expression | int | float | Decimal
            The subexpression or pattern that should be contained literally.
        """

    def get_all_symbols(self, include_function_symbols: bool = True) -> Sequence[Expression]:
        """
        Get all symbols in the current expression, optionally including function symbols.
        The symbols are sorted in Symbolica's internal ordering.

        Parameters
        ----------
        include_function_symbols: bool
            Whether function symbols should be included in the collected symbol set.
        """

    def get_all_indeterminates(self, enter_functions: bool = True) -> Sequence[Expression]:
        """
        Get all symbols and functions in the current expression, optionally including function symbols.
        The symbols are sorted in Symbolica's internal ordering.

        Parameters
        ----------
        enter_functions: bool
            Whether function arguments should be traversed when collecting indeterminates.
        """

    def to_float(self, decimal_prec: int = 16) -> Expression:
        """
        Convert all coefficients and built-in functions to floats with a given precision `decimal_prec`.
        The precision of floating point coefficients in the input will be truncated to `decimal_prec`.

        Parameters
        ----------
        decimal_prec: int
            The decimal precision used during numerical evaluation.
        """

    def rationalize(self, relative_error: float = 0.01) -> Expression:
        """
        Map all floating point and rational coefficients to the best rational approximation
        in the interval `[self*(1-relative_error),self*(1+relative_error)]`.

        Parameters
        ----------
        relative_error: float
            The maximum relative error used when converting floating-point input to a rational number.
        """

    def req_len(self, min_length: int, max_length: int | None) -> PatternRestriction:
        """
        Create a pattern restriction based on the wildcard length before downcasting.

        Parameters
        ----------
        min_length: int
            The minimum required match length.
        max_length: int | None
            The maximum allowed match length.
        """

    def req_tag(self, tag: str) -> PatternRestriction:
        """
        Create a pattern restriction based on the tag of a matched variable or function.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x', tags=['a', 'b'])
        >>> x_ = S('x_')
        >>> e = x.replace(x_, 1, x_.req_tag('b'))
        >>> print(e)  # 1
        Parameters
        ----------
        tag: str
            The tag to test or require.
        """

    def req_attr(self, tag: SymbolAttribute) -> PatternRestriction:
        """
        Create a pattern restriction based on the attributes of a matched variable or function.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('f', is_linear=True)
        >>> x_ = S('x_')
        >>> print(E('f(x)').replace(E('x_(x)'), 1, ~S('x_').req_attr(SymbolAttribute.Linear)))
        >>> print(e)  # f(x)
        Parameters
        ----------
        tag: SymbolAttribute
            The tag to test or require.
        """

    def req_type(self, atom_type: AtomType) -> PatternRestriction:
        """
        Create a pattern restriction that tests the type of the atom.

        Examples
        --------
        >>> from symbolica import *
        >>> x, x_ = S('x', 'x_')
        >>> f = S('f')
        >>> e = f(x)*f(2)*f(f(3))
        >>> e = e.replace(f(x_), 1, x_.req_type(AtomType.Num))
        >>> print(e)  # f(x)*f(1)
        Parameters
        ----------
        atom_type: AtomType
            The atom type to test or require.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the expression is of a certain type.

        Parameters
        ----------
        atom_type: AtomType
            The atom type to test or require.
        """

    def req_contains(self, a: Expression) -> PatternRestriction:
        """
        Create a pattern restriction that filters for expressions that contain `a`.

        Parameters
        ----------
        a: Expression
            The expression that must occur inside the match.
        """

    def req_lit(self) -> PatternRestriction:
        """
        Create a pattern restriction that treats the wildcard as a literal variable,
        so that it only matches to itself.
        """

    def req(
        self,
        filter_fn: Callable[[Expression], bool | Condition],
    ) -> PatternRestriction:
        """
        Create a new pattern restriction that calls the function `filter_fn` with the matched
        atom that should return a boolean. If true, the pattern matches.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace(f(x_), 1, x_.req(lambda m: m == 2 or m == 3))

        Parameters
        ----------
        filter_fn: Callable[[Expression], bool | Condition]
            A callback that filters partially constructed graphs.
        """

    def req_cmp(
        self,
        other: Expression | int | float | complex | Decimal,
        cmp_fn: Callable[[Expression, Expression], bool | Condition],
    ) -> PatternRestriction:
        """
        Create a new pattern restriction that calls the function `cmp_fn` with another the matched
        atom and the match atom of the `other` wildcard that should return a boolean. If true, the pattern matches.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, y_ = S('x_', 'y_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace(f(x_)*f(y_), 1, x_.req_cmp(y_, lambda m1, m2: m1 + m2 == 4))

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        cmp_fn: Callable[[Expression, Expression], bool | Condition]
            The comparison callback applied to the matched values.
        """

    def req_lt(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace(f(x_), 1, x_.req_lt(2))

        Parameters
        ----------
        num: Expression | int | float | complex | Decimal
            The value that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_gt(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is greater than a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace(f(x_), 1, x_.req_gt(2))

        Parameters
        ----------
        num: Expression | int | float | complex | Decimal
            The value that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_le(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace(f(x_), 1, x_.req_le(2))

        Parameters
        ----------
        num: Expression | int | float | complex | Decimal
            The value that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_ge(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
        If the matched wildcard is not a number, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> e = e.replace(f(x_), 1, x_.req_ge(2))

        Parameters
        ----------
        num: Expression | int | float | complex | Decimal
            The value that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_cmp_lt(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is smaller than another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, y_ = S('x_', 'y_')
        >>> f = S('f')
        >>> e = f(1,2)
        >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_lt(y_))

        Parameters
        ----------
        num: Expression
            The expression that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_cmp_gt(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is greater than another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, y_ = S('x_', 'y_')
        >>> f = S('f')
        >>> e = f(1,2)
        >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_gt(y_))

        Parameters
        ----------
        num: Expression
            The expression that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_cmp_le(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is smaller than or equal to another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, y_ = S('x_', 'y_')
        >>> f = S('f')
        >>> e = f(1,2)
        >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_le(y_))

        Parameters
        ----------
        num: Expression
            The expression that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def req_cmp_ge(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """
        Create a pattern restriction that passes when the wildcard is greater than or equal to another wildcard.
        If the matched wildcards are not a numbers, the pattern fails.

        When the option `cmp_any_atom` is set to `True`, this function compares atoms
        of any type. The result depends on the internal ordering and may change between
        different Symbolica versions.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, y_ = S('x_', 'y_')
        >>> f = S('f')
        >>> e = f(1,2)
        >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_ge(y_))

        Parameters
        ----------
        num: Expression
            The expression that the match is compared against.
        cmp_any_atom: Any
            Whether the comparison may be satisfied by any atom in the expression instead of only the whole match.
        """

    def __eq__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __neq__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __lt__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __le__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __gt__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __ge__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __iter__(self) -> Iterator[Expression]:
        """
        Create an iterator over all subexpressions of the expression.
        """

    def __getitem__(self, idx: int) -> Expression:
        """
        Get the `idx`th component of the expression.

        Parameters
        ----------
        idx: int
            The zero-based index to access.
        """

    def map(
        self,
        transformations: Transformer,
        n_cores: int | None = 1,
        stats_to_file: str | None = None,
    ) -> Expression:
        """
        Map the transformations to every term in the expression.
        The execution happens in parallel using `n_cores`.

        Examples
        --------
        >>> x, x_ = S('x', 'x_')
        >>> e = (1+x)**2
        >>> r = e.map(T().expand().replace(x, 6))
        >>> print(r)

        Parameters
        ----------
        transformations: Transformer
            The transformations to apply.
        n_cores: int, optional
            The number of CPU cores used for parallel execution.
        stats_to_file: str, optional
            If set, the output of the `stats` transformer will be written to a file in JSON format.
        """

    def set_coefficient_ring(self, vars: Sequence[Expression]) -> Expression:
        """
        Set the coefficient ring to contain the variables in the `vars` list.
        This will move all variables into a rational polynomial function.

        Parameters
        ----------
        vars : Sequence[Expression]
                A list of variables
        """

    def expand(self, var: Expression | None = None, via_poly: bool | None = None) -> Expression:
        """
        Expand the expression. Optionally, expand in `var` only. `var` can be a variable or a function.
        If it is a variable, any function with that variable name is also expanded in.
        To expand in multiple functions at the same time, wrap them in a function with the same symbol first,
        using a match and replace, and then expand in that function.

        Using `via_poly=True` may give a significant speedup for large expressions.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, f, g = S('x', 'y', 'f', 'g')
        >>> e = (f(1) + g(2))*(f(3) + (y+1)**2)
        >>> print(e.expand(f))

        yields `f(1)*f(3)+f(3)*g(2)+(1+y)^2*f(1)+(1+y)^2*g(2)`.

        Parameters
        ----------
        var: Expression | None
            The variable to expand with respect to. If omitted, expand all variables.
        via_poly: bool | None
            Whether the operation should use an intermediate polynomial representation.
        """

    def expand_num(self) -> Expression:
        """
         Distribute numbers in the expression, for example: `2*(x+y)` -> `2*x+2*y`.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = 3*(x+y)*(4*x+5*y)
        >>> print(e.expand_num())

        yields

        ```
        (3*x+3*y)*(4*x+5*y)
        ```
        """

    def collect(
        self,
        *x: Expression,
        key_map: Callable[[Expression], Expression] | None = None,
        coeff_map: Callable[[Expression], Expression] | None = None,
    ) -> Expression:
        """
        Collect terms involving the same power of the indeterminate(s) `x`.
        Return the list of key-coefficient pairs and the remainder that matched no key.

        Both the key (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` respectively.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.collect(x))  # x^2+x*(y+5)+5
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> var, coeff = S('var', 'coeff')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.collect(x, key_map=lambda x: var(x), coeff_map=lambda x: coeff(x)))

        yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.

        Parameters
        ----------
        *x: Expression
            The variable(s) or function(s) to collect terms in
        key_map
            A function to be applied to the quantity collected in
        coeff_map
            A function to be applied to the coefficient
        """

    def collect_symbol(
        self,
        x: Expression,
        key_map: Callable[[Expression], Expression] | None = None,
        coeff_map: Callable[[Expression], Expression] | None = None,
    ) -> Expression:
        """
        Collect terms involving the same power of variables or functions with the name `x`.

        Both the *key* (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` respectively.

        Examples
        --------

        >>> from symbolica import *
        >>> x, f = S('x', 'f')
        >>> e = f(1,2) + x*f(1,2)
        >>>
        >>> print(e.collect_symbol(f))  # (1+x)*f(1,2)
        Parameters
        ----------
        x: Expression
            The symbol to collect in
        key_map
            A function to be applied to the quantity collected in
        coeff_map
            A function to be applied to the coefficient
        """

    def collect_factors(self) -> Expression:
        """
        Collect common factors from (nested) sums.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('x*(x+y*x+x^2+y*(x+x^2))')
        >>> e.collect_factors()

        yields

        ```
        v1^2*(1+v1+v2+v2*(1+v1))
        ```
        """

    def collect_horner(self, vars: Sequence[Expression] | None = None) -> Expression:
        """
        Iteratively extract the minimal common powers of an indeterminate `v` for every term that contains `v`
        and continue to the next indeterminate in `variables`.
        This is a generalization of Horner's method for polynomials.

        If no variables are provided, a heuristically determined variable ordering is used
        that minimizes the number of operations.

        Examples
        --------

        >>> from symbolica import *
        >>> expr = E('v1 + v1*v2 + 2 v1*v2*v3 + v1^2 + v1^3*y + v1^4*z')
        >>> collected = expr.collect_horner([S('v1'), S('v2')])

        yields `v1*(1+v1*(1+v1*(v1*z+y))+v2*(1+2*v3))`.

        Parameters
        ----------
        vars: Sequence[Expression] | None
            The variables treated as polynomial variables, in the given order.
        """

    def collect_num(self) -> Expression:
        """
        Collect numerical factors by removing the content from additions.
        For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.

        The first argument of the addition is normalized to a positive quantity.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = (-3*x+6*y)*(2*x+2*y)
        >>> print(e.collect_num())

        yields

        ```
        -6*(x+y)*(x-2*y)
        ```
        """

    def coefficient_list(
        self, *x: Expression
    ) -> Sequence[tuple[Expression, Expression]]:
        """
        Collect terms involving the same power of `x`, where `x` are variables or functions.
        Return the list of key-coefficient pairs.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> for a in e.coefficient_list(x):
        >>>     print(a[0], a[1])

        yields
        ```
        x y+5
        x^2 1
        1 5
        ```

        Parameters
        ----------
        x: Expression
            The variables whose coefficient exponents should be listed.
        """

    def coefficient(self, x: Expression) -> Expression:
        """
        Collect terms involving the literal occurrence of `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = 5*x + x * y + x**2 + y*x**2
        >>> print(e.coefficient(x**2))

        yields

        ```
        y + 1
        ```

        Parameters
        ----------
        x: Expression
            The variable whose coefficient should be extracted.
        """

    def derivative(self, x: Expression) -> Expression:
        """
        Derive the expression w.r.t the variable `x`.

        Parameters
        ----------
        x: Expression
            The variable with respect to which to differentiate.
        """

    def series(
        self,
        x: Expression,
        expansion_point: Expression | int | float | complex | Decimal,
        depth: int,
        depth_denom: int = 1,
        depth_is_absolute: bool = True
    ) -> Series:
        """
        Series expand in `x` around `expansion_point` to depth `depth`.

        Examples
        --------

        >>> p = E('cos(x)/(x+1)')
        >>> print(p.series(S('x'), 0, 3))

        yields `-1-x-1/2*x^2-1/2*x^3+𝒪(x^4)`

        Parameters
        ----------

        x : Expression
            The variable to expand in.
        expansion_point : Expression | int | float | complex | Decimal
            The point around which to expand.
        depth : int
            The depth of the expansion.
        depth_denom : int, optional
            The denominator of the depth (for a rational depth), by default 1.
        depth_is_absolute : bool, optional
            If `True`, `depth` is the absolute depth in `x`; if `False`, `depth` is the
            relative to the lowest order encountered in the expression.
        """

    def apart(self, x: Expression | None = None) -> Expression:
        """
        Compute the partial fraction decomposition in `x`.

        If `None` is passed, the expression will be decomposed in all variables
        which involves a potentially expensive Groebner basis computation.


        Examples
        --------

        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))')
        >>> print(p.apart(S('x')))

        Multivariate partial fractioning:
        >>> p = E('(2y-x)/(y*(x+y)*(y-x))')
        >>> print(p.apart())

        yields `3/2*y^-1*(x+y)^-1+1/2*y^-1*(-x+y)^-1`

        Parameters
        ----------
        x: Expression | None
            The variable with respect to which to perform the partial-fraction decomposition.
        """

    def together(self) -> Expression:
        """
        Write the expression over a common denominator.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('v1^2/2+v1^3/v4*v2+v3/(1+v4)')
        >>> print(p.together())
        """

    def cancel(self) -> Expression:
        """
        Cancel common factors between numerators and denominators.
        Any non-canceling parts of the expression will not be rewritten.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('1+(y+1)^10*(x+1)/(x^2+2x+1)')
        >>> print(p.cancel())  # 1+(y+1)**10/(x+1)
        """

    def factor(self) -> Expression:
        """
        Factor the expression over the rationals.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(6 + x)/(7776 + 6480*x + 2160*x^2 + 360*x^3 + 30*x^4 + x^5)')
        >>> print(p.factor())  # (x+6)**-4
        """

    @overload
    def to_polynomial(self, vars: Sequence[Expression] | None = None) -> Polynomial:
        """
        Convert the expression to a polynomial, optionally, with the variable ordering specified in `vars`.
        All non-polynomial parts will be converted to new, independent variables.

        Parameters
        ----------
        vars: Sequence[Expression] | None
            The variables treated as polynomial variables, in the given order.
        """

    @overload
    def to_polynomial(self, minimal_poly: Polynomial, vars: Sequence[Expression] | None = None,
                      ) -> NumberFieldPolynomial:
        """
        Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
        All non-polynomial elements will be converted to new independent variables.

        The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
        The minimal polynomial must be a monic, irreducible univariate polynomial.

        Parameters
        ----------
        minimal_poly: Polynomial
            The minimal polynomial that defines the algebraic extension.
        vars: Sequence[Expression] | None
            The variables treated as polynomial variables, in the given order.
        """

    @overload
    def to_polynomial(self,
                      modulus: int,
                      power: tuple[int, Expression] | None = None,
                      minimal_poly: Polynomial | None = None,
                      vars: Sequence[Expression] | None = None,
                      ) -> FiniteFieldPolynomial:
        """
        Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
        All non-polynomial elements will be converted to new independent variables.

        The coefficients will be converted to finite field elements modulo `modulus`.
        If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
        `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

        If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.

        Parameters
        ----------
        modulus: int
            The modulus that defines the finite field.
        power: tuple[int, Expression] | None
            The extension degree and generator that define the finite field.
        minimal_poly: Polynomial | None
            The minimal polynomial that defines the algebraic extension.
        vars: Sequence[Expression] | None
            The variables treated as polynomial variables, in the given order.
        """

    def to_rational_polynomial(
        self,
        vars: Sequence[Expression] | None = None,
    ) -> RationalPolynomial:
        """
        Convert the expression to a rational polynomial, optionally, with the variable ordering specified in `vars`.
        The latter is useful if it is known in advance that more variables may be added in the future to the
        rational polynomial through composition with other rational polynomials.

        All non-rational polynomial parts are converted to new, independent variables.

        Examples
        --------
        >>> a = E('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
        >>> print(a)

        Parameters
        ----------
        vars: Sequence[Expression] | None
            The variables treated as polynomial variables, in the given order.
        """

    def match(
        self,
        lhs: Expression | int | float | complex | Decimal,
        cond: PatternRestriction | Condition | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
    ) -> MatchIterator:
        """
        Return an iterator over the pattern `self` matching to `lhs`.
        Restrictions on the pattern can be supplied through `cond`.

        The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
        The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
        depending on `level_is_tree_depth`.

        Examples
        --------

        >>> x, x_ = S('x','x_')
        >>> f = S('f')
        >>> e = f(x)*f(1)*f(2)*f(3)
        >>> for match in e.match(f(x_)):
        >>>    for map in match:
        >>>        print(map[0],'=', map[1])

        Parameters
        ----------
        lhs: Expression | int | float | complex | Decimal
            The expression to match against.
        cond: PatternRestriction | Condition | None
            An additional restriction that a match or replacement must satisfy.
        min_level: int
            The minimum level at which a match is allowed.
        max_level: int | None
            The maximum level at which a match is allowed.
        level_range: tuple[int, int | None] | None
            The `(min_level, max_level)` range in which matches are allowed.
        level_is_tree_depth: bool
            Whether levels should be measured by tree depth instead of function nesting.
        partial: bool
            Whether matches are allowed inside larger expressions instead of only at the top level.
        allow_new_wildcards_on_rhs: bool
            Whether wildcards that appear only on the right-hand side are allowed.
        """

    def matches(
        self,
        lhs: Expression | int | float | complex | Decimal,
        cond: PatternRestriction | Condition | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
    ) -> Condition:
        """
        Test whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.

        Examples
        --------

        >>> f = S('f')
        >>> if f(1).matches(f(2)):
        >>>    print('match')

        Parameters
        ----------
        lhs: Expression | int | float | complex | Decimal
            The expression to match against.
        cond: PatternRestriction | Condition | None
            An additional restriction that a match or replacement must satisfy.
        min_level: int
            The minimum level at which a match is allowed.
        max_level: int | None
            The maximum level at which a match is allowed.
        level_range: tuple[int, int | None] | None
            The `(min_level, max_level)` range in which matches are allowed.
        level_is_tree_depth: bool
            Whether levels should be measured by tree depth instead of function nesting.
        partial: bool
            Whether matches are allowed inside larger expressions instead of only at the top level.
        allow_new_wildcards_on_rhs: bool
            Whether wildcards that appear only on the right-hand side are allowed.
        """

    def replace_iter(
        self,
        lhs: Expression | int | float | complex | Decimal,
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
        cond: PatternRestriction | Condition | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
    ) -> ReplaceIterator:
        """
        Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
        Restrictions on pattern can be supplied through `cond`.

        Examples
        --------

        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(1)*f(2)*f(3)
        >>> for r in e.replace_iter(f(x_), f(x_ + 1)):
        >>>     print(r)

        Yields:
        ```
        f(2)*f(2)*f(3)
        f(1)*f(3)*f(3)
        f(1)*f(2)*f(4)
        ```

        Parameters
        ----------
        lhs:
            The pattern to match.
        rhs:
            The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
        cond:
            Conditions on the pattern.
        min_level: int, optional
            The minimum level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        max_level: int | None, optional
            The maximum level at which the pattern is allowed to match. `None` means no maximum.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
            Prefer setting `min_level` and `max_level` directly over `level_range`, as this argument will be deprecated in the future.
        level_is_tree_depth: bool, optional
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        partial: bool, optional
            If set to `True`, allow the pattern to match to a part of a term. For example, with `partial=True`, the pattern `x+y` matches to `x+2+y`.
        allow_new_wildcards_on_rhs: bool, optional
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        """

    def replace(
        self,
        pattern: Expression | int | float | complex | Decimal,
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
        cond: PatternRestriction | Condition | None = None,
        non_greedy_wildcards: Sequence[Expression] | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
        rhs_cache_size: int | None = None,
        repeat: bool = False,
        once: bool = False,
        bottom_up: bool = False,
        nested: bool = False,
    ) -> Expression:
        """
        Replace all subexpressions matching the pattern `pattern` by the right-hand side `rhs`.
        The right-hand side can be an expression with wildcards, a held expression (see :meth:`Expression.hold`) or
        a function that maps a dictionary of wildcards to an expression.

        Examples
        --------

        >>> x, w1_, w2_ = S('x','w1_','w2_')
        >>> f = S('f')
        >>> e = f(3,x)
        >>> r = e.replace(f(w1_,w2_), f(w1_ - 1, w2_**2), w1_ >= 1)
        >>> print(r)

        Parameters
        ----------
        self:
            The expression to match and replace on.
        pattern:
            The pattern to match.
        rhs:
            The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
        cond: PatternRestriction | Condition, optional
            Conditions on the pattern.
        non_greedy_wildcards: Sequence[Expression], optional
            Wildcards that try to match as little as possible.
        min_level: int, optional
            The minimum level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        max_level: int | None, optional
            The maximum level at which the pattern is allowed to match. `None` means no maximum.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
            Prefer setting `min_level` and `max_level` directly over `level_range`, as this argument will be deprecated in the future.
        level_is_tree_depth: bool, optional
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        partial: bool, optional
            If set to `True`, allow the pattern to match to a part of a term. For example, with `partial=True`, the pattern `x+y` matches to `x+2+y`.
        allow_new_wildcards_on_rhs: bool, optional
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        rhs_cache_size: int, optional
            Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
            **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
        repeat: bool, optional
            If set to `True`, the entire operation will be repeated until there are no more matches.
        once: bool, optional
            If set to `True`, only the first match will be replaced, instead of all non-overlapping matches.
        bottom_up: bool, optional
            Replace deepest nested matches first instead of replacing the outermost matches first.
            For example, replacing `f(x_)` with `x_^2` in `f(f(x))` would yield `f(x)^2` with the default settings and `f(x^2)` with bottom-up replacement.
        nested: bool, optional
            Replace nested matches, starting from the deepest first and acting on the result of that replacement.
            For example, replacing `f(x_)` with `x_^2` in `f(f(x))` would yield `f(x)^2` with the default settings and `f(x^2)^2` with nested replacement.
        """

    def replace_multiple(self, replacements: Sequence[Replacement],  repeat: bool = False, once: bool = False, bottom_up: bool = False, nested: bool = False) -> Expression:
        """
        Replace all atoms matching the patterns. See `replace` for more information.

        The entire operation can be repeated until there are no more matches using `repeat=True`.

        Examples
        --------

        >>> x, y, f = S('x', 'y', 'f')
        >>> e = f(x,y)
        >>> r = e.replace_multiple([Replacement(x, y), Replacement(y, x)])
        >>> print(r)  # f(y,x)

        Parameters
        ----------
        replacements: Sequence[Replacement]
            The list of replacements to apply.
        repeat: bool, optional
            If set to `True`, the entire operation will be repeated until there are no more matches.
        """

    def replace_wildcards(self, replacements: dict[Expression, Expression]) -> Expression:
        """
        Replace all wildcards in the expression with the corresponding values in `replacements`.
        This function can be used to substitute the result from (see :meth:`Expression.match`)
        into its pattern.

        Examples
        --------

        >>> x, x_, f= S('x', 'x_', 'f')
        >>> e = 1 + x + f(2)
        >>> p = f(x_)
        >>> r = next(e.match(p))
        >>> p.replace_wildcards(r)
        f(2)

        Parameters
        ----------
        replacements: dict[Expression, Expression]
            A map of wildcards to their replacements.
        """

    @classmethod
    def solve_linear_system(
        _cls,
        system: Sequence[Expression],
        variables: Sequence[Expression],
        warn_if_underdetermined: bool = True,
    ) -> Sequence[Expression]:
        """
        Solve a linear system in the variables `variables`, where each expression
        in the system is understood to yield 0.

        If the system is underdetermined, a partial solution is returned
        where each bound variable is a linear combination of the free
        variables. The free variables are chosen such that they have the highest index in the `vars` list.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, c = S('x', 'y', 'c')
        >>> f = S('f')
        >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
        >>> print('x =', x_r, ', y =', y_r)

        Parameters
        ----------
        system: Sequence[Expression]
            The equations or polynomials that define the system.
        variables: Sequence[Expression]
            The variables to solve for, in order.
        warn_if_underdetermined: bool
            Whether to warn when the system is underdetermined.
        """

    @overload
    def nsolve(
        self,
        variable: Expression,
        init: float,
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> float:
        """
        Find the root of an expression in `x` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = E("x^2-2").nsolve(E("x"), 1., 0.0001, 1000000)

        Parameters
        ----------
        variable: Expression
            The variable to solve for.
        init: float
            The initial guess for Newton's method.
        prec: float
            The numerical tolerance for the Newton iteration.
        max_iter: int
            The maximum number of Newton iterations.
        """

    @overload
    def nsolve(
        self,
        variable: Expression,
        init: Decimal,
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Decimal:
        """
        Find the root of an expression in `x` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = E("x^2-2").nsolve(E("x"),
                      Decimal("1.000000000000000000000000000000000000000000000000000000000000000000000000"), 1e-74, 1000000)

        Parameters
        ----------
        variable: Expression
            The variable to solve for.
        init: Decimal
            The initial guess for Newton's method.
        prec: float
            The numerical tolerance for the Newton iteration.
        max_iter: int
            The maximum number of Newton iterations.
        """

    @overload
    @classmethod
    def nsolve_system(
        _cls,
        system: Sequence[Expression],
        variables: Sequence[Expression],
        init: Sequence[float],
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Sequence[float]:
        """
        Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = Expression.nsolve_system([E("5x^2+x*y^2+sin(2y)^2 - 2"), E("exp(2x-y)+4y - 3")], [S("x"), S("y")],
                             [1., 1.], 1e-20, 1000000)

        Parameters
        ----------
        system: Sequence[Expression]
            The equations or polynomials that define the system.
        variables: Sequence[Expression]
            The variables to solve for, in order.
        init: Sequence[float]
            The initial guess for Newton's method.
        prec: float
            The numerical tolerance for the Newton iteration.
        max_iter: int
            The maximum number of Newton iterations.
        """

    @overload
    @classmethod
    def nsolve_system(
        _cls,
        system: Sequence[Expression],
        variables: Sequence[Expression],
        init: Sequence[Decimal],
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Sequence[Decimal]:
        """
        Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = Expression.nsolve_system([E("5x^2+x*y^2+sin(2y)^2 - 2"), E("exp(2x-y)+4y - 3")], [S("x"), S("y")],
                             [Decimal("1.00000000000000000"), Decimal("1.00000000000000000")], 1e-20, 1000000)

        Parameters
        ----------
        system: Sequence[Expression]
            The equations or polynomials that define the system.
        variables: Sequence[Expression]
            The variables to solve for, in order.
        init: Sequence[Decimal]
            The initial guess for Newton's method.
        prec: float
            The numerical tolerance for the Newton iteration.
        max_iter: int
            The maximum number of Newton iterations.
        """

    def evaluate(
        self, constants: dict[Expression, float], functions: dict[Expression, Callable[[Sequence[float]], float]]
    ) -> float:
        """
        Evaluate the expression, using a map of all the variables and
        user functions to a float.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> f = S('f')
        >>> e = E('cos(x)')*3 + f(x,2)
        >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))

        Parameters
        ----------
        constants: dict[Expression, float]
            The constant substitutions applied during evaluation.
        functions: dict[Expression, Callable[[Sequence[float]], float]]
            The function callback table used during evaluation.
        """

    def evaluate_with_prec(
        self,
        constants: dict[Expression, float | str | Decimal],
        functions: dict[Expression, Callable[[Sequence[Decimal]], float | str | Decimal]],
        decimal_digit_precision: int
    ) -> Decimal:
        """
        Evaluate the expression, using a map of all the constants and
        user functions using arbitrary precision arithmetic.
        The user has to specify the number of decimal digits of precision
        and provide all input numbers as floats, strings or `Decimal`.

        Examples
        --------
        >>> from symbolica import *
        >>> from decimal import Decimal, getcontext
        >>> x = S('x', 'f')
        >>> e = E('cos(x)')*3 + f(x, 2)
        >>> getcontext().prec = 100
        >>> a = e.evaluate_with_prec({x: Decimal('1.123456789')}, {
        >>>                         f: lambda args: args[0] + args[1]}, 100)

        Parameters
        ----------
        constants: dict[Expression, float | str | Decimal]
            The constant substitutions applied during evaluation.
        functions: dict[Expression, Callable[[Sequence[Decimal]], float | str | Decimal]]
            The function callback table used during evaluation.
        decimal_digit_precision: int
            The decimal precision used for arbitrary-precision evaluation.
        """

    def evaluate_complex(
        self, constants: dict[Expression, float | complex], functions: dict[Expression, Callable[[Sequence[complex]], float | complex]]
    ) -> complex:
        """
        Evaluate the expression, using a map of all the variables and
        user functions to a complex number.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = E('sqrt(x)')*y
        >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))

        Parameters
        ----------
        constants: dict[Expression, float | complex]
            The constant substitutions applied during evaluation.
        functions: dict[Expression, Callable[[Sequence[complex]], float | complex]]
            The function callback table used during evaluation.
        """

    def evaluator(
        self,
        constants: dict[Expression, Expression],
        functions: dict[tuple[Expression, str, Sequence[Expression]], Expression],
        params: Sequence[Expression],
        iterations: int = 1,
        cpe_iterations: int | None = None,
        n_cores: int = 4,
        verbose: bool = False,
        jit_compile: bool = True,
        direct_translation: bool = True,
        max_horner_scheme_variables: int = 500,
        max_common_pair_cache_entries: int = 1000000,
        max_common_pair_distance: int = 100,
        external_functions: dict[tuple[Expression, str], Callable[[
            Sequence[float | complex]], float | complex]] | None = None,
        conditionals: Sequence[Expression] | None = None,
    ) -> Evaluator:
        """
        Create an evaluator that can evaluate (nested) expressions in an optimized fashion.
        All constants and functions should be provided as dictionaries, where the function
        dictionary has a key `(name, printable name, arguments)` and the value is the function
        body. For example the function `f(x,y)=x^2+y` should be provided as
        `{(f, "f", (x, y)): x**2 + y}`. All free parameters should be provided in the `params` list.

        Additionally, external functions can be registered that will call a Python function.

        If `KeyboardInterrupt` is triggered during the optimization, the optimization will stop and will yield the
        current best result.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, z, pi, f, g = S(
        >>>     'x', 'y', 'z', 'pi', 'f', 'g')
        >>>
        >>> e1 = E("x + pi + cos(x) + f(g(x+1),x*2)")
        >>> fd = E("y^2 + z^2*y^2")
        >>> gd = E("y + 5")
        >>>
        >>> ev = e1.evaluator({pi: Expression.num(22)/7},
        >>>              {(f, "f", (y, z)): fd, (g, "g", (y, )): gd}, [x])
        >>> res = ev.evaluate([[1.], [2.], [3.]])  # evaluate at x=1, x=2, x=3
        >>> print(res)


        Define an external function:

        >>> E("f(x)").evaluator({}, {}, [S("x")],
                    external_functions={(S("f"), "F"): lambda args: args[0]**2 + 1})

        Define an conditional function which yields `x+1` when `y != 0` and `x+2` when `y == 0`:

        >>> E("if(y, x + 1, x + 2)").evaluator({}, {}, [S("x"), S("y")], conditional=[S("if")])

        Parameters
        ----------
        constants: dict[Expression, Expression]
            A map of expressions to constants. The constants should be numerical expressions.
        functions: dict[tuple[Expression, str, Sequence[Expression]], Expression]
            A dictionary of functions. The key is a tuple of the function name, printable name and the argument variables.
            The value is the function body. If the function name entry contains arguments, these are considered tags.
        params: Sequence[Expression]
            A list of free parameters.
        iterations: int, optional
            The number of Horner schemes to try.
        cpe_iterations: int | None, optional
            The number of CPE iterations to perform. The number if unbounded if `None`.
        n_cores: int, optional
            The number of CPU cores used for the optimization.
        verbose: bool, optional
            Print the progress of the optimization.
        jit_compile: bool, optional
            Just-in-time compile the evaluator upon first use with SymJIT. This can provide
            significant performance improvements.
        direct_translation: bool, optional
            If set to `True`, the optimized expression will be directly constructed from atoms without building a tree.
        max_horner_scheme_variables: int, optional
            The maximum number of variables in a Horner scheme.
        max_common_pair_cache_entries: int, optional
            The maximum number of entries in the common pair cache.
        max_common_pair_distance: int, optional
            The maximum distance between common pairs. Used when clearing cache entries.
        external_functions: dict[tuple[Expression, str], Callable[[Sequence[float | complex]], float | complex]] | None
            A dictionary of external functions that can be called during evaluation.
            The key is a tuple of the function symbol and a printable function name.
            The value is a callable that takes a list of arguments and returns a float or complex number.
            This is useful for functions that are not defined in Symbolica but are available in Python.
        conditionals: Sequence[Expression] | None, optional
            A list of conditional functions. These functions should take three argument: a condition that is tested for
            inequality with 0, the true branch and the false branch.
        """

    @classmethod
    def evaluator_multiple(
        _cls,
        exprs: Sequence[Expression],
        constants: dict[Expression, Expression],
        functions: dict[tuple[Expression, str, Sequence[Expression]], Expression],
        params: Sequence[Expression],
        iterations: int = 1,
        cpe_iterations: int | None = None,
        n_cores: int = 4,
        verbose: bool = False,
        jit_compile: bool = True,
        direct_translation: bool = True,
        max_horner_scheme_variables: int = 500,
        max_common_pair_cache_entries: int = 1000000,
        max_common_pair_distance: int = 100,
        external_functions: dict[tuple[Expression, str], Callable[[
            Sequence[float | complex]], float | complex]] | None = None,
        conditionals: Sequence[Expression] | None = None,
    ) -> Evaluator:
        """
        Create an evaluator that can jointly evaluate (nested) expressions in an optimized fashion.
        See `Expression.evaluator()` for more information.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> e1 = E("x^2 + 1")
        >>> e2 = E("x^2 + 2")
        >>> ev = Expression.evaluator_multiple([e1, e2], {}, {}, [x])

        will recycle the `x^2`

        Parameters
        ----------
        exprs: Sequence[Expression]
            The expressions to compile into a joint evaluator.
        constants: dict[Expression, Expression]
            The symbolic substitutions applied to constant symbols when building the evaluator.
        functions: dict[tuple[Expression, str, Sequence[Expression]], Expression]
            The symbolic function implementations applied when building the evaluator.
        params: Sequence[Expression]
            The evaluator parameters, in input order.
        iterations: int
            The number of optimization passes to run.
        cpe_iterations: int | None
            The number of common subexpression elimination iterations to perform.
        n_cores: int
            The number of CPU cores used for parallel optimization.
        verbose: bool
            Whether verbose output should be enabled.
        jit_compile: bool
            Whether JIT compilation should be enabled.
        direct_translation: bool
            Whether to prefer direct translation when compiling the evaluator.
        max_horner_scheme_variables: int
            The maximum number of variables considered for Horner-scheme optimization.
        max_common_pair_cache_entries: int
            The maximum number of common-subexpression pairs to cache.
        max_common_pair_distance: int
            The maximum distance between factors when searching for common pairs.
        external_functions: dict[tuple[Expression, str], Callable[[ Sequence[float | complex]], float | complex]] | None
            The external functions to register.
        conditionals: Sequence[Expression] | None
            Expressions that should be treated as conditional branches during evaluator construction.
        """

    def canonize_tensors(self,
                         contracted_indices: Sequence[tuple[Expression | int, Expression | int]]) -> tuple[Expression, list[tuple[Expression, Expression]], list[tuple[Expression, Expression]]]:
        """
        Canonize (products of) tensors in the expression by relabeling repeated indices.
        The tensors must be written as functions, with its indices as the arguments.
        Subexpressions, constants and open indices are supported.

        If the contracted indices are distinguishable (for example in their dimension),
        you can provide a group marker as the second element in the tuple of the index
        specification.
        This makes sure that an index will not be renamed to an index from a different group.

        Returns the canonical expression, as well as the external indices and ordered dummy indices
        appearing in the canonical expression.

        Examples
        --------
        >>> g = S('g', is_symmetric=True)
        >>> fc = S('fc', is_cyclesymmetric=True)
        >>> mu1, mu2, mu3, mu4, k1 = S('mu1', 'mu2', 'mu3', 'mu4', 'k1')
        >>> e = g(mu2, mu3)*fc(mu4, mu2, k1, mu4, k1, mu3)
        >>> (r, external, dummy) = e.canonize_tensors([(mu1, 0), (mu2, 0), (mu3, 0), (mu4, 0)])
        >>> print(r)

        yields `g(mu1, mu2)*fc(mu1, mu3, mu2, k1, mu3, k1)`.

        Parameters
        ----------
        contracted_indices: Sequence[tuple[Expression | int, Expression | int]]
            The index patterns that should be treated as contracted, optionally grouped by a marker.
        """


class Replacement:
    """A replacement of a pattern by a right-hand side."""

    def __new__(
            cls,
            pattern: Expression | int | float | complex | Decimal,
            rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
            cond: PatternRestriction | Condition | None = None,
            non_greedy_wildcards: Sequence[Expression] | None = None,
            min_level: int = 0,
            max_level: int | None = None,
            level_range: tuple[int, int | None] | None = None,
            level_is_tree_depth: bool = False,
            partial: bool = True,
            allow_new_wildcards_on_rhs: bool = False,
            rhs_cache_size: int | None = None) -> Replacement:
        """
        Create a new replacement. See `replace` for more information.

        Parameters
        ----------
        pattern: Expression | int | float | complex | Decimal
            The left-hand-side pattern to match.
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal
            The right-hand-side operand.
        cond: PatternRestriction | Condition | None
            An additional restriction that a match or replacement must satisfy.
        non_greedy_wildcards: Sequence[Expression] | None
            Wildcards that should be matched non-greedily.
        min_level: int
            The minimum level at which a match is allowed.
        max_level: int | None
            The maximum level at which a match is allowed.
        level_range: tuple[int, int | None] | None
            The `(min_level, max_level)` range in which matches are allowed.
        level_is_tree_depth: bool
            Whether levels should be measured by tree depth instead of function nesting.
        partial: bool
            Whether matches are allowed inside larger expressions instead of only at the top level.
        allow_new_wildcards_on_rhs: bool
            Whether wildcards that appear only on the right-hand side are allowed.
        rhs_cache_size: int | None
            The cache size for memoizing right-hand-side evaluations.
        """


class PatternRestriction:
    """A restriction on wildcards."""

    def __and__(self, other: PatternRestriction) -> PatternRestriction:
        """
        Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold).

        Parameters
        ----------
        other: PatternRestriction
            The other operand to combine or compare with.
        """

    def __or__(self, other: PatternRestriction) -> PatternRestriction:
        """
        Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold).

        Parameters
        ----------
        other: PatternRestriction
            The other operand to combine or compare with.
        """

    def __invert__(self) -> PatternRestriction:
        """
        Create a new pattern restriction that takes the logical 'not' of the current restriction.
        """

    @classmethod
    def req_matches(_cls, match_fn: Callable[[dict[Expression, Expression]], int]) -> PatternRestriction:
        """
        Create a pattern restriction based on the current matched variables.
        `match_fn` is a Python function that takes a dictionary of wildcards and their matched values
        and should return an integer. If the integer is less than 0, the restriction is false.
        If the integer is 0, the restriction is inconclusive.
        If the integer is greater than 0, the restriction is true.

        If your pattern restriction cannot decide if it holds since not all the required variables
        have been matched, it should return inclusive (0).

        Examples
        --------
        >>> from symbolica import *
        >>> f, x_, y_, z_ = S('f', 'x_', 'y_', 'z_')
        >>>
        >>> def filter(m: dict[Expression, Expression]) -> int:
        >>>    if x_ in m and y_ in m:
        >>>        if m[x_] > m[y_]:
        >>>            return -1  # no match
        >>>        if z_ in m:
        >>>            if m[y_] > m[z_]:
        >>>                return -1
        >>>            return 1  # match
        >>>
        >>>    return 0  # inconclusive
        >>>
        >>>
        >>> e = f(1, 2, 3).replace(f(x_, y_, z_), 1,
        >>>         PatternRestriction.req_matches(filter))

        Parameters
        ----------
        match_fn: Callable[[dict[Expression, Expression]], int]
            The callback evaluated on each match.
        """


class Condition:
    """Relations that evaluate to booleans"""

    def eval(self) -> bool:
        """
        Evaluate the condition.
        """

    def __repr__(self) -> str:
        """
        Return a string representation of the condition.
        """

    def __str__(self) -> str:
        """
        Return a string representation of the condition.
        """

    def __bool__(self) -> bool:
        """
        Return the boolean value of the condition.
        """

    def __and__(self, other:  Condition) -> Condition:
        """
        Create a condition that is the logical and operation between two conditions (i.e., both should hold).

        Parameters
        ----------
        other: Condition
            The other operand to combine or compare with.
        """

    def __or__(self, other:  Condition) -> Condition:
        """
        Create a condition that is the logical 'or' operation between two conditions (i.e., at least one of the two should hold).

        Parameters
        ----------
        other: Condition
            The other operand to combine or compare with.
        """

    def __invert__(self) -> Condition:
        """
        Create a condition that takes the logical 'not' of the current condition.
        """

    def to_req(self) -> PatternRestriction:
        """
        Convert the condition to a pattern restriction.
        """


class CompareOp:
    """One of the following comparison operators: `<`,`>`,`<=`,`>=`,`==`,`!=`."""


class HeldExpression:
    def __call__(self) -> Expression:
        """
        Execute a bound transformer. If the transformer is unbound,
        you can call it with an expression as an argument.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> e = (x+1)**5
        >>> e = e.hold(T().expand())()
        >>> print(e)
        """

    def __eq__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __neq__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __lt__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __le__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __gt__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __ge__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the transformed expression is of a certain type.

        Parameters
        ----------
        atom_type: AtomType
            The atom type to test or require.
        """

    def contains(self, element: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Create a transformer that checks if the expression contains the given `element`.

        Parameters
        ----------
        element: HeldExpression | Expression | int | float | complex | Decimal
            The element that should be contained in the expression.
        """

    def matches(
        self,
        lhs: Expression | int | float | complex | Decimal,
        cond: PatternRestriction | Condition | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
    ) -> Condition:
        """
        Create a transformer that tests whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.

        Parameters
        ----------
        lhs: Expression | int | float | complex | Decimal
            The expression to match against.
        cond: PatternRestriction | Condition | None
            An additional restriction that a match or replacement must satisfy.
        min_level: int
            The minimum level at which a match is allowed.
        max_level: int | None
            The maximum level at which a match is allowed.
        level_range: tuple[int, int | None] | None
            The `(min_level, max_level)` range in which matches are allowed.
        level_is_tree_depth: bool
            Whether levels should be measured by tree depth instead of function nesting.
        partial: bool
            Whether matches are allowed inside larger expressions instead of only at the top level.
        allow_new_wildcards_on_rhs: bool
            Whether wildcards that appear only on the right-hand side are allowed.
        """

    def __add__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __radd__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __sub__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Subtract `other` from this transformer, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __rsub__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Subtract this transformer from `other`, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __mul__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __rmul__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __truediv__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Divide this transformer by `other`, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __rtruediv__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Divide `other` by this transformer, returning the result.

        Parameters
        ----------
        other: HeldExpression | Expression | int | float | complex | Decimal
            The other operand to combine or compare with.
        """

    def __pow__(self, exp: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Take `self` to power `exp`, returning the result.

        Parameters
        ----------
        exp: HeldExpression | Expression | int | float | complex | Decimal
            The exponent.
        """

    def __rpow__(self, base: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Take `base` to power `self`, returning the result.

        Parameters
        ----------
        base: HeldExpression | Expression | int | float | complex | Decimal
            The base expression.
        """

    def __xor__(self, a: Any) -> HeldExpression:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.

        Parameters
        ----------
        a: Any
            The operand passed with `^`; use `**` for exponentiation instead.
        """

    def __rxor__(self, a: Any) -> HeldExpression:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.

        Parameters
        ----------
        a: Any
            The operand passed with `^`; use `**` for exponentiation instead.
        """

    def __neg__(self) -> HeldExpression:
        """
        Negate the current transformer, returning the result.
        """


class Transformer:
    """Operations that transform an expression."""

    def __new__(_cls) -> Transformer:
        """
        Create a new transformer for a term provided by `Expression.map`.
        """

    def __call__(self, expr: Expression | int | float | complex | Decimal, stats_to_file: str | None = None) -> Expression:
        """
        Execute an unbound transformer on the given expression. If the transformer
        is bound, use `execute()` instead.

        Examples
        --------
        >>> x = S('x')
        >>> e = T().expand()((1+x)**2)

        Parameters
        ----------
        expr: Expression
            The expression to transform.
        stats_to_file: str, optional
            If set, the output of the `stats` transformer will be written to a file in JSON format.
        """

    def if_then(self, condition: Condition, if_block: Transformer, else_block: Transformer | None = None) -> Transformer:
        """
        Evaluate the condition and apply the `if_block` if the condition is true, otherwise apply the `else_block`.
        The expression that is the input of the transformer is the input for the condition, the `if_block` and the `else_block`.

        Examples
        --------
        >>> t = T().map_terms(T().if_then(T().contains(x), T().print()))
        >>> t(x + y + 4)

        prints `x`.

        Parameters
        ----------
        condition: Condition
            The condition to evaluate.
        if_block: Transformer
            The transformer to apply when the condition is true.
        else_block: Transformer | None
            The transformer to apply when the condition is false.
        """

    def if_changed(self, condition: Transformer, if_block: Transformer, else_block: Transformer | None = None) -> Transformer:
        """
        Execute the `condition` transformer. If the result of the `condition` transformer is different from the input expression,
        apply the `if_block`, otherwise apply the `else_block`. The input expression of the `if_block` is the output
        of the `condition` transformer.

        Examples
        --------
        >>> t = T().map_terms(T().if_changed(T().replace(x, y), T().print()))
        >>> print(t(x + y + 4))

        prints
        ```
        y
        2*y+4
        ```

        Parameters
        ----------
        condition: Transformer
            The condition to evaluate.
        if_block: Transformer
            The transformer to apply when the condition is true.
        else_block: Transformer | None
            The transformer to apply when the condition is false.
        """

    def break_chain(self) -> Transformer:
        """
        Break the current chain and all higher-level chains containing `if` transformers.

        Examples
        --------
        >>> from symbolica import *
        >>> t = T().map_terms(T().repeat(
        >>>     T().replace(y, 4),
        >>>     T().if_changed(T().replace(x, y),
        >>>                 T().break_chain()),
        >>>     T().print()  # print of y is never reached
        >>> ))
        >>> print(t(x))
        """

    def expand(self, var: Expression | None = None, via_poly: bool | None = None) -> Transformer:
        """
        Create a transformer that expands products and powers. Optionally, expand in `var` only.

        Using `via_poly=True` may give a significant speedup for large expressions.

        Examples
        --------
        >>> from symbolica import *
        >>> x, x_ = S('x', 'x_')
        >>> f = S('f')
        >>> e = f((x+1)**2).replace(f(x_), x_.hold(T().expand()))
        >>> print(e)

        Parameters
        ----------
        var: Expression | None
            The variable to expand with respect to. If omitted, expand all variables.
        via_poly: bool | None
            Whether the operation should use an intermediate polynomial representation.
        """

    def expand_num(self) -> Expression:
        """
        Create a transformer that distributes numbers in the expression, for example: `2*(x+y)` -> `2*x+2*y`.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = 3*(x+y)*(4*x+5*y)
        >>> print(T().expand_num()(e))

        yields

        ```
        (3*x+3*y)*(4*x+5*y)
        ```
        """

    def prod(self) -> Transformer:
        """
        Create a transformer that computes the product of a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(2,3).replace(f(x__), x__.hold(T().prod()))
        >>> print(e)
        """

    def sum(self) -> Transformer:
        """
        Create a transformer that computes the sum of a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(2,3).replace(f(x__), x__.hold(T().sum()))
        >>> print(e)
        """

    def nargs(self, only_for_arg_fun: bool = False) -> Transformer:
        """
        Create a transformer that returns the number of arguments.
        If the argument is not a function, return 0.

        If `only_for_arg_fun` is `True`, only count the number of arguments
        in the `arg()` function and return 1 if the input is not `arg`.
        This is useful for obtaining the length of a range during pattern matching.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(2,3,4).replace(f(x__), x__.hold(T().nargs()))
        >>> print(e)

        Parameters
        ----------
        only_for_arg_fun: bool
            Whether the transformer should only count arguments of `arg(...)`.
        """

    def sort(self) -> Transformer:
        """
        Create a transformer that sorts a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(3,2,1).replace(f(x__), x__.hold(T().sort()))
        >>> print(e)
        """

    def cycle_symmetrize(self) -> Transformer:
        """
        Create a transformer that cycle-symmetrizes a function.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x__')
        >>> f = S('f')
        >>> e = f(1,2,4,1,2,3).replace(f(x__), x_.hold(T().cycle_symmetrize()))
        >>> print(e)  # f(1,2,3,1,2,4)
        """

    def deduplicate(self) -> Transformer:
        """
        Create a transformer that removes elements from a list if they occur
        earlier in the list as well.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(1,2,1,2).replace(f(x__), x__.hold(T().deduplicate()))
        >>> print(e)  # f(1,2)
        """

    def from_coeff(self) -> Transformer:
        """
        Create a transformer that extracts a rational polynomial from a coefficient.

        Examples
        --------
        >>> from symbolica import *, Function
        >>> e = Function.COEFF((x^2+1)/y^2).hold(T().from_coeff())
        >>> print(e)
        """

    def split(self) -> Transformer:
        """
        Create a transformer that split a sum or product into a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x, x__ = S('x', 'x__')
        >>> f = S('f')
        >>> e = (x + 1).replace(x__, f(x_.hold(T().split())))
        >>> print(e)
        """

    def linearize(self, symbols: Sequence[Expression] | None) -> Transformer:
        """
        Create a transformer that linearizes a function, optionally extracting `symbols`
        as well.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y, z, w, f, x__ = S('x', 'y', 'z', 'w', 'f', 'x__')
        >>> e = f(x+y, 4*z*w+3).replace(f(x__), f(x__).hold(T().linearize([z])))
        >>> print(e)  # f(x,3)+f(y,3)+4*z*f(x,w)+4*z*f(y,w)
        Parameters
        ----------
        symbols: Sequence[Expression] | None
            The symbols to linearize with respect to.
        """

    def partitions(
        self,
        bins: Sequence[tuple[Transformer | Expression, int]],
        fill_last: bool = False,
        repeat: bool = False,
    ) -> Transformer:
        """
        Create a transformer that partitions a list of arguments into named bins of a given length,
        returning all partitions and their multiplicity.

        If the unordered list `elements` is larger than the bins, setting the flag `fill_last`
        will add all remaining elements to the last bin.

        Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
        if possible.

        Note that the functions names to be provided for the bin names must be generated through `Expression.var`.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, f_id, g_id = S('x__', 'f', 'g')
        >>> f = S('f')
        >>> e = f(1,2,1,3).replace(f(x_), x_.hold(T().partitions([(f_id, 2), (g_id, 1), (f_id, 1)])))
        >>> print(e)

        yields:
        ```
        2*f(1)*f(1,2)*g(3)+2*f(1)*f(1,3)*g(2)+2*f(1)*f(2,3)*g(1)+f(2)*f(1,1)*g(3)+2*f(2)*f(1,3)*g(1)+f(3)*f(1,1)*g(2)+2*f(3)*f(1,2)*g(1)
        ```

        Parameters
        ----------
        bins: Sequence[tuple[Transformer | Expression, int]]
            The output bins and their required lengths.
        fill_last: bool
            Whether any remaining elements should be placed in the last bin.
        repeat: bool
            Whether the transformation should be applied repeatedly until it no longer changes the expression.
        """

    def permutations(self, function_name: Transformer | Expression) -> Transformer:
        """
        Create a transformer that generates all permutations of a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x_, f_id = S('x__', 'f')
        >>> f = S('f')
        >>> e = f(1,2,1,2).replace(f(x_), x_.hold(T().permutations(f_id)))
        >>> print(e)

        yields:
        ```
        4*f(1,1,2,2)+4*f(1,2,1,2)+4*f(1,2,2,1)+4*f(2,1,1,2)+4*f(2,1,2,1)+4*f(2,2,1,1)
        ```

        Parameters
        ----------
        function_name: Transformer | Expression
            The function symbol used to wrap each generated permutation.
        """

    def map(self, f: Callable[[Expression], Expression | int | float | complex | Decimal]) -> Transformer:
        """
        Create a transformer that applies a Python function.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(2).replace(f(x_), x_.hold(T().map(lambda r: r**2)))
        >>> print(e)

        Parameters
        ----------
        f: Callable[[Expression], Expression | int | float | complex | Decimal]
            The callback or function to apply.
        """

    def map_terms(self, *transformers: Transformer, n_cores: int = 1) -> Transformer:
        """
        Map a chain of transformer over the terms of the expression, optionally using multiple cores.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> t = T().map_terms(T().print(), n_cores=2)
        >>> e = t(x + y)

        Parameters
        ----------
        transformers: Transformer
            The transformers to chain or apply.
        n_cores: int
            The number of CPU cores used to map over terms.
        """

    def for_each(self, *transformers: Transformer) -> Transformer:
        """
        Create a transformer that applies a transformer chain to every argument of the `arg()` function.
        If the input is not `arg()`, the transformer is applied to the input.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> f = S('f')
        >>> t = T().split().for_each(T().map(f))
        >>> e = t(1+x)

        Parameters
        ----------
        transformers: Transformer
            The transformers to chain or apply.
        """

    def check_interrupt(self) -> Transformer:
        """
        Create a transformer that checks for a Python interrupt,
        such as ctrl-c and aborts the current transformer.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> t = T().replace(f(x_), f(x_ + 1)).check_interrupt()
        >>> t(f(10))
        """

    def repeat(self, *transformers: Transformer) -> Transformer:
        """
        Create a transformer that repeatedly executes the arguments in order
        until there are no more changes.
        The output from one transformer is inserted into the next.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> t = T().repeat(
        >>>     T().expand(),
        >>>     T().replace(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
        >>> )
        >>> e = t(f(5))

        Parameters
        ----------
        transformers: Transformer
            The transformers to chain or apply.
        """

    def chain(self, *transformers: Transformer) -> Transformer:
        """
        Chain several transformers. `chain(A,B,C)` is the same as `A.B.C`,
        where `A`, `B`, `C` are transformers.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = E("f(5)")
        >>> t = T().chain(
        >>>     T().expand(),
        >>>     T().replace(f(x_), f(5))
        >>> )
        >>> e = t(f(5))

        Parameters
        ----------
        transformers: Transformer
            The transformers to chain or apply.
        """

    def derivative(self, x: HeldExpression | Expression) -> Transformer:
        """
        Create a transformer that derives `self` w.r.t the variable `x`.

        Parameters
        ----------
        x: HeldExpression | Expression
            The variable with respect to which to differentiate.
        """

    def set_coefficient_ring(self, vars: Sequence[Expression]) -> Transformer:
        """
        Create a transformer that sets the coefficient ring to contain the variables in the `vars` list.
        This will move all variables into a rational polynomial function.

        Parameters
        ----------
        vars : Sequence[Expression]
                A list of variables
        """

    def collect(
        self,
        *x: Expression,
        key_map: Transformer | None = None,
        coeff_map: Transformer | None = None,
    ) -> Transformer:
        """
        Create a transformer that collects terms involving the same power of the indeterminate(s) `x`.
        Return the list of key-coefficient pairs and the remainder that matched no key.

        Both the key (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` transformers respectively.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = 5*x + x * y + x**2 + 5
        >>>
        >>> print(e.hold(T().collect(x).execute()))  # x^2+x*(y+5)+5
        >>> from symbolica import *
        >>> x, y, x_, var, coeff = S('x', 'y', 'x_', 'var', 'coeff')
        >>> e = 5*x + x * y + x**2 + 5
        >>> print(e.collect(x, key_map=T().replace(x_, var(x_)),
                coeff_map=T().replace(x_, coeff(x_))))

        yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.

        Parameters
        ----------
        *x: Expression
            The variable(s) or function(s) to collect terms in
        key_map: Transformer
            A transformer to be applied to the quantity collected in
        coeff_map: Transformer
            A transformer to be applied to the coefficient
        """

    def collect_symbol(
        self,
        x: Expression,
        key_map: Callable[[Expression], Expression] | None = None,
        coeff_map: Callable[[Expression], Expression] | None = None,
    ) -> Transformer:
        """
        Create a transformer that collects terms involving the same power of variables or functions with the name `x`.

        Both the *key* (the quantity collected in) and its coefficient can be mapped using
        `key_map` and `coeff_map` respectively.

        Examples
        --------

        >>> from symbolica import *
        >>> x, f = S('x', 'f')
        >>> e = f(1,2) + x*f(1,2)
        >>>
        >>> print(T().collect_symbol(x)(e))  # (1+x)*f(1,2)
        Parameters
        ----------
        x: Expression
            The symbol to collect in
        key_map: Transformer
            A transformer to be applied to the quantity collected in
        coeff_map: Transformer
            A transformer to be applied to the coefficient
        """

    def collect_factors(self) -> Transformer:
        """
        Create a transformer that collects common factors from (nested) sums.

        Examples
        --------

        >>> from symbolica import *
        >>> t = T().collect_factors()
        >>> t(E('x*(x+y*x+x^2+y*(x+x^2))'))

        yields

        ```
        v1^2*(1+v1+v2+v2*(1+v1))
        ```
        """

    def collect_horner(self, vars: Sequence[Expression] | None = None) -> Transformer:
        """
        Create a transformer that iteratively extracts the minimal common powers of an indeterminate `v` for every term that contains `v`
        and continues to the next indeterminate in `variables`.
        This is a generalization of Horner's method for polynomials.

        If no variables are provided, a heuristically determined variable ordering is used
        that minimizes the number of operations.

        Examples
        --------

        >>> from symbolica import *
        >>> expr = E('v1 + v1*v2 + 2 v1*v2*v3 + v1^2 + v1^3*y + v1^4*z')
        >>> collected = expr.hold(T().collect_horner([S('v1'), S('v2')]))()

        yields `v1*(1+v1*(1+v1*(v1*z+y))+v2*(1+2*v3))`.

        Parameters
        ----------
        vars: Sequence[Expression] | None
            The variables treated as polynomial variables, in the given order.
        """

    def collect_num(self) -> Transformer:
        """
        Create a transformer that collects numerical factors by removing the content from additions.
        For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.

        The first argument of the addition is normalized to a positive quantity.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = (-3*x+6*y)*(2*x+2*y)
        >>> print(T().collect_num()(e))

        yields

        ```
        -6*(x+y)*(x-2*y)
        ```
        """

    def conjugate(self) -> Transformer:
        """
        Complex conjugate the expression.
        """

    def coefficient(self, x: Expression) -> Transformer:
        """
        Create a transformer that collects terms involving the literal occurrence of `x`.

        Parameters
        ----------
        x: Expression
            The variable whose coefficient should be extracted.
        """

    def apart(self, x: Expression) -> Transformer:
        """
        Create a transformer that computes the partial fraction decomposition in `x`.

        Parameters
        ----------
        x: Expression
            The variable with respect to which to perform the partial-fraction decomposition.
        """

    def together(self) -> Transformer:
        """
        Create a transformer that writes the expression over a common denominator.
        """

    def cancel(self) -> Transformer:
        """
        Create a transformer that cancels common factors between numerators and denominators.
        Any non-canceling parts of the expression will not be rewritten.
        """

    def factor(self) -> Transformer:
        """
        Create a transformer that factors the expression over the rationals.
        """

    def series(
        self,
        x: Expression,
        expansion_point: Expression | int | float | complex | Decimal,
        depth: int,
        depth_denom: int = 1,
        depth_is_absolute: bool = True
    ) -> Transformer:
        """
        Create a transformer that series expands in `x` around `expansion_point` to depth `depth`.

        Examples
        -------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> f = S('f')
        >>>
        >>> e = 2* x**2 * y + f(x)
        >>> e = e.series(x, 0, 2)
        >>>
        >>> print(e)

        yields `f(0)+x*der(1,f(0))+1/2*x^2*(der(2,f(0))+4*y)`.

        Parameters
        ----------
        x: Expression
            The variable around which the series is expanded.
        expansion_point: Expression | int | float | complex | Decimal
            The point around which the series should be expanded.
        depth: int
            The numerator of the expansion depth.
        depth_denom: int
            The denominator of the fractional expansion depth.
        depth_is_absolute: bool
            Whether the requested depth is measured as an absolute order instead of relative to the leading term.
        """

    def replace(
        self,
        pat: HeldExpression | Expression | int | float | complex | Decimal,
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
        cond: PatternRestriction | Condition | None = None,
        non_greedy_wildcards: Sequence[Expression] | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
        rhs_cache_size: int | None = None,
        once: bool = False,
        bottom_up: bool = False,
        nested: bool = False
    ) -> Transformer:
        """
        Create a transformer that replaces all subexpressions matching the pattern `pat` by the right-hand side `rhs`.

        Examples
        --------

        >>> x, w1_, w2_ = S('x','w1_','w2_')
        >>> f = S('f')
        >>> t = T().replace(f(w1_, w2_), f(w1_ - 1, w2_**2), w1_ >= 1)
        >>> r = t(f(3,x))
        >>> print(r)

        Parameters
        ----------
        pat:
            The pattern to match.
        rhs:
            The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
        cond:
            Conditions on the pattern.
        non_greedy_wildcards:
            Wildcards that try to match as little as possible.
        cond: PatternRestriction | Condition, optional
            Conditions on the pattern.
        non_greedy_wildcards: Sequence[Expression], optional
            Wildcards that try to match as little as possible.
        min_level: int, optional
            The minimum level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        max_level: int | None, optional
            The maximum level at which the pattern is allowed to match. `None` means no maximum.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
            Prefer setting `min_level` and `max_level` directly over `level_range`, as this argument will be deprecated in the future.
        level_is_tree_depth: bool, optional
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        partial: bool, optional
            If set to `True`, allow the pattern to match to a part of a term. For example, with `partial=True`, the pattern `x+y` matches to `x+2+y`.
        allow_new_wildcards_on_rhs: bool, optional
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        rhs_cache_size: int, optional
            Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
            **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
        once: bool, optional
            If set to `True`, only the first match will be replaced, instead of all non-overlapping matches.
        bottom_up: bool, optional
            Replace deepest nested matches first instead of replacing the outermost matches first.
            For example, replacing `f(x_)` with `x_^2` in `f(f(x))` would yield `f(x)^2` with the default settings and `f(x^2)` with bottom-up replacement.
        nested: bool, optional
            Replace nested matches, starting from the deepest first and acting on the result of that replacement.
            For example, replacing `f(x_)` with `x_^2` in `f(f(x))` would yield `f(x)^2` with the default settings and `f(x^2)^2` with nested replacement.
        """

    def replace_multiple(self, replacements: Sequence[Replacement], once: bool = False, bottom_up: bool = False, nested: bool = False) -> Transformer:
        """
        Create a transformer that replaces all atoms matching the patterns. See `replace` for more information.

        Examples
        --------

        >>> x, y, f = S('x', 'y', 'f')
        >>> t = T().replace_multiple([Replacement(x, y), Replacement(y, x)])
        >>> r = t(f(x,y))
        >>> print(r)

        Parameters
        ----------
        replacements: Sequence[Replacement]
            The replacements to apply.
        once: bool
            Whether only the first matching replacement should be applied.
        bottom_up: bool
            Whether the transformation should traverse the expression from leaves to root.
        nested: bool
            Whether matches created by replacements may be matched again inside the same pass.
        """

    def print(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        bracket_level_colors: Sequence[int] | None = [
            244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = None,
        custom_print_mode: int | None = None,
    ) -> Transformer:
        """
        Create a transformer that prints the expression.

        Examples
        --------
        >>> T().print(terms_on_new_line = True)

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        terms_on_new_line: bool
            Whether wrapped output should place terms on separate lines.
        color_top_level_sum: bool
            Whether top-level sums should be colorized.
        color_builtin_symbols: bool
            Whether built-in symbols should be colorized.
        bracket_level_colors: Sequence[int] | None
            The colors assigned to successive nested bracket levels.
        print_ring: bool
            Whether the coefficient ring should be included in the printed output.
        symmetric_representation_for_finite_field: bool
            Whether finite-field elements should be printed using symmetric representatives.
        explicit_rational_polynomial: bool
            Whether rational polynomials should be printed explicitly as numerator and denominator.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def stats(
        self,
        tag: str,
        transformer: Transformer,
        color_medium_change_threshold: float | None = 10.,
        color_large_change_threshold: float | None = 100.,
    ) -> Transformer:
        """
        Print statistics of a transformer, tagging it with `tag`.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> t = T().stats('replace', T().replace(f(x_), 1)).execute()
        >>> t(eE('f(5)'))

        yields
        ```
        Stats for replace:
            In  │ 1 │  10.00 B │
            Out │ 1 │   3.00 B │ ⧗ 40.15µs
        ```

        Parameters
        ----------
        tag: str
            The tag to test or require.
        transformer: Transformer
            The transformer whose statistics should be recorded.
        color_medium_change_threshold: float | None
            The percentage change threshold that should be highlighted as a medium change.
        color_large_change_threshold: float | None
            The percentage change threshold that should be highlighted as a large change.
        """

    def __eq__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers.

        Parameters
        ----------
        other: Transformer | Expression | int | float | Decimal
            The other operand to combine or compare with.
        """

    def __neq__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers.

        Parameters
        ----------
        other: Transformer | Expression | int | float | Decimal
            The other operand to combine or compare with.
        """

    def __lt__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Transformer | Expression | int | float | Decimal
            The other operand to combine or compare with.
        """

    def __le__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Transformer | Expression | int | float | Decimal
            The other operand to combine or compare with.
        """

    def __gt__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Transformer | Expression | int | float | Decimal
            The other operand to combine or compare with.
        """

    def __ge__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.

        Parameters
        ----------
        other: Transformer | Expression | int | float | Decimal
            The other operand to combine or compare with.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the transformed expression is of a certain type.

        Parameters
        ----------
        atom_type: AtomType
            The atom type to test or require.
        """

    def contains(self, element: Transformer | HeldExpression | Expression | int | float | Decimal) -> Condition:
        """
        Create a transformer that checks if the expression contains the given `element`.

        Parameters
        ----------
        element: Transformer | HeldExpression | Expression | int | float | Decimal
            The element that should be contained in the expression.
        """

    def matches(
        self,
        lhs: HeldExpression | Expression | int | float | Decimal,
        cond: PatternRestriction | Condition | None = None,
        min_level: int = 0,
        max_level: int | None = None,
        level_range: tuple[int, int | None] | None = None,
        level_is_tree_depth: bool = False,
        partial: bool = True,
        allow_new_wildcards_on_rhs: bool = False,
    ) -> Condition:
        """
        Create a transformer that tests whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.

        Parameters
        ----------
        lhs: HeldExpression | Expression | int | float | Decimal
            The expression to match against.
        cond: PatternRestriction | Condition | None
            An additional restriction that a match or replacement must satisfy.
        min_level: int
            The minimum level at which a match is allowed.
        max_level: int | None
            The maximum level at which a match is allowed.
        level_range: tuple[int, int | None] | None
            The `(min_level, max_level)` range in which matches are allowed.
        level_is_tree_depth: bool
            Whether levels should be measured by tree depth instead of function nesting.
        partial: bool
            Whether matches are allowed inside larger expressions instead of only at the top level.
        allow_new_wildcards_on_rhs: bool
            Whether wildcards that appear only on the right-hand side are allowed.
        """


class Series:
    """
    A series expansion class.

    Supports standard arithmetic operations, such
    as addition and multiplication.

    Examples
    --------
    >>> x = S('x')
    >>> s = E("(1-cos(x))/sin(x)").series(x, 0, 4) * x
    >>> print(s)
    """

    def __getitem__(self, expr: Expression | int) -> Expression:
        """
        Get the coefficient of the term with exponent `exp`

        Parameters
        ----------
        expr: Expression | int
            The expression to operate on.
        """

    def get_coefficient(self, exp:  Expression | int) -> Expression:
        """
        Get the coefficient of the term with exponent `exp`.  Alternatively, use `series[exp]`.

        Parameters
        ----------
        exp: Expression | int
            The exponent whose coefficient should be returned.
        """

    def __iter__(self) -> Iterator[tuple[Expression, Expression]]:
        """
        Iterate over the terms of the series, yielding pairs of exponent and coefficient.
        """

    def __str__(self) -> str:
        """
        Print the series in a human-readable format.
        """

    def to_latex(self) -> str:
        """
        Convert the series into a LaTeX string.
        """

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        bracket_level_colors: Sequence[int] | None = [
            244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        precision: int | None = None,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = None,
        custom_print_mode: int | None = None,
    ) -> str:
        """
        Convert the series into a human-readable string.

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        terms_on_new_line: bool
            Whether wrapped output should place terms on separate lines.
        color_top_level_sum: bool
            Whether top-level sums should be colorized.
        color_builtin_symbols: bool
            Whether built-in symbols should be colorized.
        bracket_level_colors: Sequence[int] | None
            The colors assigned to successive nested bracket levels.
        print_ring: bool
            Whether the coefficient ring should be included in the printed output.
        symmetric_representation_for_finite_field: bool
            Whether finite-field elements should be printed using symmetric representatives.
        explicit_rational_polynomial: bool
            Whether rational polynomials should be printed explicitly as numerator and denominator.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        precision: int | None
            The decimal precision used when printing numeric coefficients.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def __add__(self, other: Series | Expression) -> Series:
        """
        Add another series or expression to this series, returning the result.

        Parameters
        ----------
        other: Series | Expression
            The other operand to combine or compare with.
        """

    def __radd__(self, other: Expression) -> Series:
        """
        Add two series together, returning the result.

        Parameters
        ----------
        other: Expression
            The other operand to combine or compare with.
        """

    def __sub__(self, other: Series | Expression) -> Series:
        """
        Subtract `other` from `self`, returning the result.

        Parameters
        ----------
        other: Series | Expression
            The other operand to combine or compare with.
        """

    def __rsub__(self, other: Expression) -> Series:
        """
        Subtract `self` from `other`, returning the result.

        Parameters
        ----------
        other: Expression
            The other operand to combine or compare with.
        """

    def __mul__(self, other: Series | Expression) -> Series:
        """
        Multiply another series or expression to this series, returning the result.

        Parameters
        ----------
        other: Series | Expression
            The other operand to combine or compare with.
        """

    def __rmul__(self, other: Expression) -> Series:
        """
        Multiply two series together, returning the result.

        Parameters
        ----------
        other: Expression
            The other operand to combine or compare with.
        """

    def __truediv__(self, other: Series | Expression) -> Series:
        """
        Divide `self` by `other`, returning the result.

        Parameters
        ----------
        other: Series | Expression
            The other operand to combine or compare with.
        """

    def __rtruediv__(self, other: Expression) -> Series:
        """
        Divide `other` by `self`, returning the result.

        Parameters
        ----------
        other: Expression
            The other operand to combine or compare with.
        """

    def __pow__(self, exp: int) -> Series:
        """
        Raise the series to the power of `exp`, returning the result.

        Parameters
        ----------
        exp: int
            The exponent.
        """

    def __neg__(self) -> Series:
        """
        Negate the series.
        """

    def sin(self) -> Series:
        """
        Compute the sine of the series, returning the result.
        """

    def cos(self) -> Series:
        """
        Compute the cosine of the series, returning the result.
        """

    def exp(self) -> Series:
        """
        Compute the exponential of the series, returning the result.
        """

    def log(self) -> Series:
        """
        Compute the natural logarithm of the series, returning the result.
        """

    def pow(self, num: int, den: int) -> Series:
        """
        Raise the series to the power of `num/den`, returning the result.

        Parameters
        ----------
        num: int
            The numerator of the rational exponent.
        den: int
            The denominator of the rational exponent.
        """

    def spow(self, exp: Series) -> Series:
        """
        Raise the series to the power of `exp`, returning the result.

        Parameters
        ----------
        exp: Series
            The series exponent.
        """

    def shift(self, e: int) -> Series:
        """
        Shift the series by `e` units of the ramification.

        Parameters
        ----------
        e: int
            The shift measured in units of the series ramification.
        """

    def get_ramification(self) -> int:
        """
        Get the ramification.
        """

    def get_trailing_exponent(self) -> tuple[int, int]:
        """
        Get the trailing exponent; the exponent of the first non-zero term.
        """

    def get_relative_order(self) -> tuple[int, int]:
        """
        Get the relative order.
        """

    def get_absolute_order(self) -> tuple[int, int]:
        """
        Get the absolute order.
        """

    def to_expression(self) -> Expression:
        """
        Convert the series to an expression
        """


class TermStreamer:
    """
    A term streamer that can handle large expressions, by
    streaming terms to and from disk.
    """

    def __new__(_cls, path: str | None = None,
                max_mem_bytes: int | None = None,
                n_cores: int | None = None) -> TermStreamer:
        """
        Create a new term streamer with a given path for its files,
        the maximum size of the memory buffer and the number of cores.

        Parameters
        ----------
        path: str | None
            The directory used for the streamer's temporary files.
        max_mem_bytes: int | None
            The maximum in-memory buffer size in bytes.
        n_cores: int | None
            The number of CPU cores used for streaming operations.
        """

    def __add__(self, other: TermStreamer) -> TermStreamer:
        """
        Add two term streamers together, returning the result.

        Parameters
        ----------
        other: TermStreamer
            The other operand to combine or compare with.
        """

    def __iadd__(self, other: TermStreamer) -> None:
        """
        Add another term streamer to this one.

        Parameters
        ----------
        other: TermStreamer
            The other operand to combine or compare with.
        """

    def clear(self) -> None:
        """
        Clear the term streamer.
        """

    def load(self, filename: str, conflict_fn: Callable[[str], str] | None = None) -> int:
        """
        Load terms and their state from a binary file into the term streamer. The number of terms loaded is returned.

        The state will be merged with the current one. If a symbol has conflicting attributes, the conflict
        can be resolved using the renaming function `conflict_fn`.

        A term stream can be exported using `TermStreamer.save`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        conflict_fn: Callable[[str], str] | None
            A callback that resolves symbol conflicts during loading.
        """

    def save(self, filename: str, compression_level: int = 9) -> None:
        """
        Export terms and their state to a binary file.
        The resulting file can be read back using `TermStreamer.load` or
        by using `Expression.load`. In the latter case, the whole term stream will be read into memory
        as a single expression.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        compression_level: int
            The compression level for serialized output.
        """

    def get_byte_size(self) -> int:
        """
        Get the byte size of the term stream.
        """

    def get_num_terms(self) -> int:
        """
        Get the number of terms in the stream.
        """

    def fits_in_memory(self) -> bool:
        """
        Check if the term stream fits in memory.
        """

    def push(self, expr: Expression) -> None:
        """
        Push an expression to the term stream.

        Parameters
        ----------
        expr: Expression
            The expression to operate on.
        """

    def normalize(self) -> None:
        """
        Sort and fuse all terms in the stream.
        """

    def to_expression(self) -> Expression:
        """
        Convert the term stream into an expression. This may exceed the available memory.
        """

    def map(self, f: Transformer, stats_to_file: str | None = None) -> TermStreamer:
        """
        Apply a transformer to all terms in the stream.

        Parameters
        ----------
        f: Transformer
            The transformer to apply.
        stats_to_file: str, optional
            If set, the output of the `stats` transformer will be written to a file in JSON format.
        """

    def map_single_thread(self, f: Transformer, stats_to_file: str | None = None) -> TermStreamer:
        """
        Apply a transformer to all terms in the stream using a single thread.

        Parameters
        ----------
        f: Transformer
            The transformer to apply.
        stats_to_file: str, optional
            If set, the output of the `stats` transformer will be written to a file in JSON format.
        """


class MatchIterator:
    """An iterator over matches."""

    def __iter__(self) -> MatchIterator:
        """
        Create the iterator.
        """

    def __next__(self) -> dict[Expression, Expression]:
        """
        Return the next match.
        """


class ReplaceIterator:
    """An iterator over single replacements."""

    def __iter__(self) -> ReplaceIterator:
        """
        Create the iterator.
        """

    def __next__(self) -> Expression:
        """
        Return the next replacement.
        """


class Polynomial:
    """A Symbolica polynomial with rational coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], default_namespace: str | None = None) -> Polynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'])

        Parameters
        ----------
        input: str
            The input value.
        vars: Sequence[str]
            The variables treated as polynomial variables, in the given order.
        default_namespace: str | None
            The namespace assumed for unqualified symbols during parsing.

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> Polynomial:
        """
        Copy the polynomial.
        """

    def __str__(self) -> str:
        """
        Print the polynomial in a human-readable format.
        """

    def to_latex(self) -> str:
        """
        Convert the polynomial into a LaTeX string.
        """

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        bracket_level_colors: Sequence[int] | None = [
            244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        precision: int | None = None,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = None,
        custom_print_mode: int | None = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        terms_on_new_line: bool
            Whether wrapped output should place terms on separate lines.
        color_top_level_sum: bool
            Whether top-level sums should be colorized.
        color_builtin_symbols: bool
            Whether built-in symbols should be colorized.
        bracket_level_colors: Sequence[int] | None
            The colors assigned to successive nested bracket levels.
        print_ring: bool
            Whether the coefficient ring should be included in the printed output.
        symmetric_representation_for_finite_field: bool
            Whether finite-field elements should be printed using symmetric representatives.
        explicit_rational_polynomial: bool
            Whether rational polynomials should be printed explicitly as numerator and denominator.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        precision: int | None
            The decimal precision used when printing numeric coefficients.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def nterms(self) -> int:
        """
        Get the number of terms in the polynomial.
        """

    def get_variables(self) -> Sequence[Expression]:
        """
        Get the list of variables in the internal ordering of the polynomial.
        """

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are not equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __lt__(self, rhs: int) -> bool:
        """
        Check if the polynomial is less than an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __le__(self, rhs: int) -> bool:
        """
        Check if the polynomial is less than or equal to an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __gt__(self, rhs: int) -> bool:
        """
        Check if the polynomial is greater than an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __ge__(self, rhs: int) -> bool:
        """
        Check if the polynomial is greater than or equal to an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __add__(self, rhs: Polynomial | int) -> Polynomial:
        """
        Add two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __sub__(self, rhs: Polynomial | int) -> Polynomial:
        """
        Subtract polynomials `rhs` from `self`, returning the result.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __mul__(self, rhs: Polynomial | int) -> Polynomial:
        """
        Multiply two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __radd__(self, rhs: Polynomial | int) -> Polynomial:
        """
        Add two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __rsub__(self, rhs: Polynomial | int) -> Polynomial:
        """
        Subtract polynomials `self` from `rhs`, returning the result.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __rmul__(self, rhs: Polynomial | int) -> Polynomial:
        """
        Multiply two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """
        Divide the polynomial `self` by `rhs`, rounding down, returning the result.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def __truediv__(self, rhs: Polynomial) -> Polynomial:
        """
        Divide the polynomial `self` by `rhs` if possible, returning the result.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def quot_rem(self, rhs: Polynomial) -> tuple[Polynomial, Polynomial]:
        """
        Divide `self` by `rhs`, returning the quotient and remainder.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def __mod__(self, rhs: Polynomial) -> Polynomial:
        """
        Compute the remainder of the division of `self` by `rhs`.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def __neg__(self) -> Polynomial:
        """
        Negate the polynomial.
        """

    def __pow__(self, exp: int) -> Polynomial:
        """
        Raise the polynomial to the power of `exp`, returning the result.

        Parameters
        ----------
        exp: int
            The exponent.
        """

    def __contains__(self, var: Expression) -> bool:
        """
        Check if the polynomial contains the given variable.

        Parameters
        ----------
        var: Expression
            The variable whose presence should be tested.
        """

    def contains(self, var: Expression) -> bool:
        """
        Check if the polynomial contains the given variable.

        Parameters
        ----------
        var: Expression
            The variable whose presence should be tested.
        """

    def degree(self, var: Expression) -> int:
        """
        Get the degree of the polynomial in `var`.

        Parameters
        ----------
        var: Expression
            The variable whose degree should be returned.
        """

    def reorder(self, vars: Sequence[Expression]) -> None:
        """
        Reorder the polynomial in-place to use the given variable order.

        Parameters
        ----------
        vars: Sequence[Expression]
            The variables treated as polynomial variables, in the given order.
        """

    def gcd(self, *rhs: Polynomial) -> Polynomial:
        """
        Compute the greatest common divisor (GCD) of two or more polynomials.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def extended_gcd(self, rhs: Polynomial) -> tuple[Polynomial, Polynomial, Polynomial]:
        """
        Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
        such that `self * s + rhs * t = gcd(self, rhs)`.

        Examples
        --------

        >>> from symbolica import *
        >>> E('(1+x)(20+x)').to_polynomial().extended_gcd(E('x^2+2').to_polynomial())

        yields `(1, 1/67-7/402*x, 47/134+7/402*x)`.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def resultant(self, rhs: Polynomial, var: Expression) -> Polynomial:
        """
        Compute the resultant of two polynomials with respect to the variable `var`.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        var: Expression
            The variable with respect to which the resultant is computed.
        """

    def to_finite_field(self, prime: int) -> FiniteFieldPolynomial:
        """
        Convert the coefficients of the polynomial to a finite field with prime `prime`.

        Parameters
        ----------
        prime: int
            The prime modulus of the target finite field.
        """

    def isolate_roots(self, refine: float | Decimal | None = None) -> list[tuple[Expression, Expression, int]]:
        """
        Isolate the real roots of the polynomial. The result is a list of intervals with rational bounds that contain exactly one root,
        and the multiplicity of that root. Optionally, the intervals can be refined to a given precision.

        Examples
        --------
        >>> from symbolica import *
        >>> p = E('2016+5808*x+5452*x^2+1178*x^3+-753*x^4+-232*x^5+41*x^6').to_polynomial()
        >>> for a, b, n in p.isolate_roots():
        >>>     print('({},{}): {}'.format(a, b, n))

        yields
        ```
        (-56/45,-77/62): 1
        (-98/79,-119/96): 1
        (-119/96,-21/17): 1
        (-7/6,0): 1
        (0,6): 1
        (6,12): 1
        ```

        Parameters
        ----------
        refine: float | Decimal | None
            The optional interval refinement tolerance.
        """

    def approximate_roots(self, max_iterations: int, tolerance: float) -> list[tuple[complex, int]]:
        """
        Approximate all complex roots of a univariate polynomial, given a maximal number of iterations
        and a given tolerance. Returns the roots and their multiplicity.

        Examples
        --------

        >>> p = E('x^10+9x^7+4x^3+2x+1').to_polynomial()
        >>> for (r, m) in p.approximate_roots(1000, 1e-10):
        >>>     print(r, m)

        Parameters
        ----------
        max_iterations: int
            The maximum number of iterations for the root finder.
        tolerance: float
            The convergence tolerance for the root finder.
        """

    def factor_square_free(self) -> list[tuple[Polynomial, int]]:
        """
        Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[tuple[Polynomial, int]]:
        """
        Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> Polynomial:
        """
        Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.derivative(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to differentiate.
        """

    def integrate(self, x: Expression) -> Polynomial:
        """
        Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.integrate(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to integrate.
        """

    def content(self) -> Polynomial:
        """
        Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def primitive(self) -> Polynomial:
        """
        Get the primitive part of the polynomial, i.e., the polynomial
        with the content removed.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.primitive())  # 2*x^2+x+3
        """

    def monic(self) -> Polynomial:
        """
        Get the monic part of the polynomial, i.e., the polynomial
        divided by its leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.monic())  # x^2+1/2*x+2/3
        """

    def lcoeff(self) -> Polynomial:
        """
        Get the leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
        >>> print(p)  # 3
        """

    def coefficient_list(self, xs: Expression | Sequence[Expression] | None = None) -> list[tuple[list[int], Polynomial]]:
        """
        Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)

        Parameters
        ----------
        xs: Expression | Sequence[Expression] | None
            The variables with respect to which coefficients should be listed.
        """

    @classmethod
    def groebner_basis(_cls, system: Sequence[Polynomial], grevlex: bool = True, print_stats: bool = False) -> list[Polynomial]:
        """
        Compute the Groebner basis of a polynomial system.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        If `print_stats=True` intermediate statistics will be printed.

        Examples
        --------
        >>> basis = Polynomial.groebner_basis(
        >>>     [E("a b c d - 1").to_polynomial(),
        >>>     E("a b c + a b d + a c d + b c d").to_polynomial(),
        >>>     E("a b + b c + a d + c d").to_polynomial(),
        >>>     E("a + b + c + d").to_polynomial()],
        >>>     grevlex=True,
        >>>     print_stats=True
        >>> )
        >>> for p in basis:
        >>>     print(p)

        Parameters
        ----------
        system: Sequence[Polynomial]
            The equations or polynomials that define the system.
        grevlex: bool
            Whether graded reverse lexicographic ordering should be used.
        print_stats: bool
            Whether Groebner basis statistics should be printed during computation.
        """

    def reduce(self, gs: Sequence[Polynomial], grevlex: bool = True) -> Polynomial:
        """
        Completely reduce the polynomial w.r.t the polynomials `gs`.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        Examples
        --------
        >>> E('y^2+x').to_polynomial().reduce([E('x').to_polynomial()])

        yields `y^2`

        Parameters
        ----------
        gs: Sequence[Polynomial]
            The polynomials that define the reducing set.
        grevlex: bool
            Whether graded reverse lexicographic ordering should be used.
        """

    def to_expression(self) -> Expression:
        """
        Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('x*y+2*x+x^2')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def evaluate(self, input: npt.ArrayLike) -> float:
        """
        Evaluate the polynomial at point `input`.

        Examples
        --------

        >>> from symbolica import *
        >>> P('x*y+2*x+x^2').evaluate([2., 3.])

        Yields `14.0`.

        Parameters
        ----------
        input: npt.ArrayLike
            The input value.
        """

    def evaluate_complex(self, input: npt.ArrayLike) -> complex:
        """
        Evaluate the polynomial at point `input` with complex input.

        Examples
        --------

        >>> from symbolica import *
        >>> P('x*y+2*x+x^2').evaluate([2+1j, 3+2j])

        Yields `11+13j`.

        Parameters
        ----------
        input: npt.ArrayLike
            The input value.
        """

    def replace(self, x: Expression, v: Polynomial | int) -> Polynomial:
        """
        Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> r = E('y+1').to_polynomial())
        >>> p.replace(x, r)

        Parameters
        ----------
        x: Expression
            The variable to replace.
        v: Polynomial | int
            The polynomial or scalar value that should replace `x`.
        """

    @classmethod
    def interpolate(_cls, x: Expression, sample_points: Sequence[Expression | int], values: Sequence[Polynomial]) -> Polynomial:
        """
        Perform Newton interpolation in the variable `x` given the sample points
        `sample_points` and the values `values`.

        Examples
        --------
        >>> x, y = S('x', 'y')
        >>> a = Polynomial.interpolate(
        >>>         x, [4, 5], [(y**2+5).to_polynomial(), (y**3).to_polynomial()])
        >>> print(a)  # 25-5*x+5*y^2-y^2*x-4*y^3+y^3*x
        Parameters
        ----------
        x: Expression
            The interpolation variable.
        sample_points: Sequence[Expression | int]
            The sample points used for interpolation.
        values: Sequence[Polynomial]
            The values associated with the sample points.
        """

    def to_number_field(self, min_poly: Polynomial) -> NumberFieldPolynomial:
        """
        Convert the coefficients of the polynomial to a number field defined by the minimal polynomial `min_poly`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = P('a').to_number_field(P('a^2-2'))
        >>> print(a * a)  # 2
        Parameters
        ----------
        min_poly: Polynomial
            The minimal polynomial that defines the algebraic extension.
        """

    def adjoin(self, min_poly: Polynomial, new_symbol: Expression | None = None) -> tuple[Polynomial, Polynomial, Polynomial]:
        """
        Adjoin the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
        is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.

        `b`  must be irreducible over `R` and `R[a]`; this is not checked.

        If `new_symbol` is provided, the variable of the new extension will be renamed to it.
        Otherwise, the variable of the new extension will be the same as that of `b`.

        Examples
        --------

        >>> from symbolica import *
        >>> sqrt2 = P('a^2-2')
        >>> sqrt23 = P('b^2-a-3')
        >>> (min_poly, rep2, rep23) = sqrt2.adjoin(sqrt23)
        >>>
        >>> # convert to number field
        >>> a = P('a^2+b').replace(S('b'), rep23).replace(S('a'), rep2).to_number_field(min_poly)

        Parameters
        ----------
        min_poly: Polynomial
            The minimal polynomial that defines the algebraic extension.
        new_symbol: Expression | None
            The symbol chosen for the adjoined generator.
        """

    def simplify_algebraic_number(self, min_poly: Polynomial) -> Polynomial:
        """
        Find the minimal polynomial for the algebraic number represented by this polynomial
        expressed in the number field defined by `minimal_poly`.

        Examples
        --------

        >>> from symbolica import *
        >>> (min_poly, rep2, rep23) = P('a^2-2').adjoin(P('b^2-3'))
        >>> rep2.simplify_algebraic_number(min_poly)

        Yields `b^2-2`.

        Parameters
        ----------
        min_poly: Polynomial
            The minimal polynomial that defines the algebraic extension.
        """


class NumberFieldPolynomial:
    """A Symbolica polynomial with rational coefficients."""

    def __copy__(self) -> NumberFieldPolynomial:
        """
        Copy the polynomial.
        """

    def __str__(self) -> str:
        """
        Print the polynomial in a human-readable format.
        """

    def to_latex(self) -> str:
        """
        Convert the polynomial into a LaTeX string.
        """

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        bracket_level_colors: Sequence[int] | None = [
            244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        precision: int | None = None,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = None,
        custom_print_mode: int | None = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldNumberFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        terms_on_new_line: bool
            Whether wrapped output should place terms on separate lines.
        color_top_level_sum: bool
            Whether top-level sums should be colorized.
        color_builtin_symbols: bool
            Whether built-in symbols should be colorized.
        bracket_level_colors: Sequence[int] | None
            The colors assigned to successive nested bracket levels.
        print_ring: bool
            Whether the coefficient ring should be included in the printed output.
        symmetric_representation_for_finite_field: bool
            Whether finite-field elements should be printed using symmetric representatives.
        explicit_rational_polynomial: bool
            Whether rational polynomials should be printed explicitly as numerator and denominator.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        precision: int | None
            The decimal precision used when printing numeric coefficients.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def nterms(self) -> int:
        """
        Get the number of terms in the polynomial.
        """

    def get_variables(self) -> Sequence[Expression]:
        """
        Get the list of variables in the internal ordering of the polynomial.
        """

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are not equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __lt__(self, rhs: int) -> bool:
        """
        Check if the polynomial is less than an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __le__(self, rhs: int) -> bool:
        """
        Check if the polynomial is less than or equal to an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __gt__(self, rhs: int) -> bool:
        """
        Check if the polynomial is greater than an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __ge__(self, rhs: int) -> bool:
        """
        Check if the polynomial is greater than or equal to an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __add__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Add two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial | int
            The right-hand-side operand.
        """

    def __sub__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Subtract polynomials `rhs` from `self`, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial | int
            The right-hand-side operand.
        """

    def __mul__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Multiply two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial | int
            The right-hand-side operand.
        """

    def __radd__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Add two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial | int
            The right-hand-side operand.
        """

    def __rsub__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Subtract polynomials `self` from `rhs`, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial | int
            The right-hand-side operand.
        """

    def __rmul__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Multiply two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial | int
            The right-hand-side operand.
        """

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """
        Divide the polynomial `self` by `rhs`, rounding down, returning the result.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def __truediv__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """
        Divide the polynomial `self` by `rhs` if possible, returning the result.

        Parameters
        ----------
        rhs: NumberFieldPolynomial
            The right-hand-side operand.
        """

    def quot_rem(self, rhs: NumberFieldPolynomial) -> tuple[NumberFieldPolynomial, NumberFieldPolynomial]:
        """
        Divide `self` by `rhs`, returning the quotient and remainder.

        Parameters
        ----------
        rhs: NumberFieldPolynomial
            The right-hand-side operand.
        """

    def __mod__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """
        Compute the remainder of the division of `self` by `rhs`.

        Parameters
        ----------
        rhs: NumberFieldPolynomial
            The right-hand-side operand.
        """

    def __neg__(self) -> NumberFieldPolynomial:
        """
        Negate the polynomial.
        """

    def __pow__(self, exp: int) -> NumberFieldPolynomial:
        """
        Raise the polynomial to the power of `exp`, returning the result.

        Parameters
        ----------
        exp: int
            The exponent.
        """

    def __contains__(self, var: Expression) -> bool:
        """
        Check if the polynomial contains the given variable.

        Parameters
        ----------
        var: Expression
            The variable whose presence should be tested.
        """

    def contains(self, var: Expression) -> bool:
        """
        Check if the polynomial contains the given variable.

        Parameters
        ----------
        var: Expression
            The variable whose presence should be tested.
        """

    def degree(self, var: Expression) -> int:
        """
        Get the degree of the polynomial in `var`.

        Parameters
        ----------
        var: Expression
            The variable whose degree should be returned.
        """

    def reorder(self, vars: Sequence[Expression]) -> None:
        """
        Reorder the polynomial in-place to use the given variable order.

        Parameters
        ----------
        vars: Sequence[Expression]
            The variables treated as polynomial variables, in the given order.
        """

    def gcd(self, *rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """
        Compute the greatest common divisor (GCD) of two or more polynomials.

        Parameters
        ----------
        rhs: NumberFieldPolynomial
            The right-hand-side operand.
        """

    def extended_gcd(self, rhs: NumberFieldPolynomial) -> tuple[NumberFieldPolynomial, NumberFieldPolynomial, NumberFieldPolynomial]:
        """
        Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
        such that `self * s + rhs * t = gcd(self, rhs)`.

        Parameters
        ----------
        rhs: NumberFieldPolynomial
            The right-hand-side operand.
        """

    def resultant(self, rhs: NumberFieldPolynomial, var: Expression) -> NumberFieldPolynomial:
        """
        Compute the resultant of two polynomials with respect to the variable `var`.

        Parameters
        ----------
        rhs: NumberFieldPolynomial
            The right-hand-side operand.
        var: Expression
            The variable with respect to which the resultant is computed.
        """

    def factor_square_free(self) -> list[tuple[NumberFieldPolynomial, int]]:
        """
        Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[tuple[NumberFieldPolynomial, int]]:
        """
        Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> NumberFieldPolynomial:
        """
        Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.derivative(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to differentiate.
        """

    def integrate(self, x: Expression) -> NumberFieldPolynomial:
        """
        Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.integrate(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to integrate.
        """

    def content(self) -> NumberFieldPolynomial:
        """
        Get the content, i.e., the GCD of the coefficients.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3x^2+6x+9').to_polynomial()
        >>> print(p.content())
        """

    def primitive(self) -> NumberFieldPolynomial:
        """
        Get the primitive part of the polynomial, i.e., the polynomial
        with the content removed.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial()
        >>> print(p.primitive())  # x^2+2*x+3
        """

    def monic(self) -> NumberFieldPolynomial:
        """
        Get the monic part of the polynomial, i.e., the polynomial
        divided by its leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.monic())  # x^2+1/2*x+2/3
        """

    def lcoeff(self) -> NumberFieldPolynomial:
        """
        Get the leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
        >>> print(p)  # 3
        """

    def coefficient_list(self, xs: Expression | Sequence[Expression] | None = None) -> list[tuple[list[int], NumberFieldPolynomial]]:
        """
        Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)

        Parameters
        ----------
        xs: Expression | Sequence[Expression] | None
            The variables with respect to which coefficients should be listed.
        """

    @classmethod
    def groebner_basis(_cls, system: list[NumberFieldPolynomial], grevlex: bool = True, print_stats: bool = False) -> list[NumberFieldPolynomial]:
        """
        Compute the Groebner basis of a polynomial system.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        If `print_stats=True` intermediate statistics will be printed.

        Parameters
        ----------
        system: list[NumberFieldPolynomial]
            The equations or polynomials that define the system.
        grevlex: bool
            Whether graded reverse lexicographic ordering should be used.
        print_stats: bool
            Whether Groebner basis statistics should be printed during computation.
        """

    def reduce(self, gs: Sequence[Polynomial], grevlex: bool = True) -> NumberFieldPolynomial:
        """
        Completely reduce the polynomial w.r.t the polynomials `gs`.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        Examples
        --------
        >>> E('y^2+x').to_polynomial().reduce([E('x').to_polynomial()])

        yields `y^2`

        Parameters
        ----------
        gs: Sequence[Polynomial]
            The polynomials that define the reducing set.
        grevlex: bool
            Whether graded reverse lexicographic ordering should be used.
        """

    def to_expression(self) -> Expression:
        """
        Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('x*y+2*x+x^2')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def replace(self, x: Expression, v: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """
        Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> r = E('y+1').to_polynomial())
        >>> p.replace(x, r)

        Parameters
        ----------
        x: Expression
            The variable to replace.
        v: NumberFieldPolynomial | int
            The polynomial or scalar value that should replace `x`.
        """

    def get_minimal_polynomial(self) -> Polynomial:
        """
        Get the minimal polynomial of the algebraic extension.
        """

    def to_polynomial(self) -> Polynomial:
        """
        Convert the number field polynomial to a rational polynomial.
        """


class FiniteFieldPolynomial:
    """A Symbolica polynomial with finite field coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], prime: int, default_namespace: str | None = None) -> FiniteFieldPolynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = FiniteFieldPolynomial.parse('18*x^2+y+y*4', ['x', 'y'], 17)

        Parameters
        ----------
        input: str
            The input value.
        vars: Sequence[str]
            The variables treated as polynomial variables, in the given order.
        prime: int
            The prime modulus of the finite field.
        default_namespace: str | None
            The namespace assumed for unqualified symbols during parsing.

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> FiniteFieldPolynomial:
        """
        Copy the polynomial.
        """

    def __str__(self) -> str:
        """
        Print the polynomial in a human-readable format.
        """

    def to_latex(self) -> str:
        """
        Convert the polynomial into a LaTeX string.
        """

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        bracket_level_colors: Sequence[int] | None = [
            244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        precision: int | None = None,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = None,
        custom_print_mode: int | None = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        terms_on_new_line: bool
            Whether wrapped output should place terms on separate lines.
        color_top_level_sum: bool
            Whether top-level sums should be colorized.
        color_builtin_symbols: bool
            Whether built-in symbols should be colorized.
        bracket_level_colors: Sequence[int] | None
            The colors assigned to successive nested bracket levels.
        print_ring: bool
            Whether the coefficient ring should be included in the printed output.
        symmetric_representation_for_finite_field: bool
            Whether finite-field elements should be printed using symmetric representatives.
        explicit_rational_polynomial: bool
            Whether rational polynomials should be printed explicitly as numerator and denominator.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        precision: int | None
            The decimal precision used when printing numeric coefficients.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def nterms(self) -> int:
        """
        Get the number of terms in the polynomial.
        """

    def get_variables(self) -> Sequence[Expression]:
        """
        Get the list of variables in the internal ordering of the polynomial.
        """

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are not equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __add__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Add two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial | int
            The right-hand-side operand.
        """

    def __sub__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Subtract polynomials `rhs` from `self`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial | int
            The right-hand-side operand.
        """

    def __mul__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Multiply two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial | int
            The right-hand-side operand.
        """

    def __radd__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Add two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial | int
            The right-hand-side operand.
        """

    def __rsub__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Subtract polynomials `self` from `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial | int
            The right-hand-side operand.
        """

    def __rmul__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Multiply two polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial | int
            The right-hand-side operand.
        """

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """
        Divide the polynomial `self` by `rhs`, rounding down, returning the result.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def __truediv__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """
        Divide the polynomial `self` by `rhs` if possible, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial
            The right-hand-side operand.
        """

    def quot_rem(self, rhs: FiniteFieldPolynomial) -> tuple[FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """
        Divide `self` by `rhs`, returning the quotient and remainder.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial
            The right-hand-side operand.
        """

    def __mod__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """
        Compute the remainder of the division of `self` by `rhs`.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial
            The right-hand-side operand.
        """

    def __neg__(self) -> FiniteFieldPolynomial:
        """
        Negate the polynomial.
        """

    def __pow__(self, exp: int) -> FiniteFieldPolynomial:
        """
        Raise the polynomial to the power of `exp`, returning the result.

        Parameters
        ----------
        exp: int
            The exponent.
        """

    def __contains__(self, var: Expression) -> bool:
        """
        Check if the polynomial contains the given variable.

        Parameters
        ----------
        var: Expression
            The variable whose presence should be tested.
        """

    def contains(self, var: Expression) -> bool:
        """
        Check if the polynomial contains the given variable.

        Parameters
        ----------
        var: Expression
            The variable whose presence should be tested.
        """

    def degree(self, var: Expression) -> int:
        """
        Get the degree of the polynomial in `var`.

        Parameters
        ----------
        var: Expression
            The variable whose degree should be returned.
        """

    def reorder(self, vars: Sequence[Expression]) -> None:
        """
        Reorder the polynomial in-place to use the given variable order.

        Parameters
        ----------
        vars: Sequence[Expression]
            The variables treated as polynomial variables, in the given order.
        """

    def gcd(self, *rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """
        Compute the greatest common divisor (GCD) of two or more polynomials.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial
            The right-hand-side operand.
        """

    def extended_gcd(self, rhs: FiniteFieldPolynomial) -> tuple[FiniteFieldPolynomial, FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """
        Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
        such that `self * s + rhs * t = gcd(self, rhs)`.

        Examples
        --------

        >>> from symbolica import *
        >>> E('(1+x)(20+x)').to_polynomial(modulus=5).extended_gcd(E('x^2+2').to_polynomial(modulus=5))

        yields `(1, 3+4*x, 3+x)`.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial
            The right-hand-side operand.
        """

    def to_integer_polynomial(self, symmetric_representation: bool = True) -> Polynomial:
        """
        Convert the polynomial to a polynomial with integer coefficients.

        Parameters
        ----------
        symmetric_representation: bool
            Whether finite-field coefficients should use symmetric integer representatives.
        """

    def resultant(self, rhs: FiniteFieldPolynomial, var: Expression) -> FiniteFieldPolynomial:
        """
        Compute the resultant of two polynomials with respect to the variable `var`.

        Parameters
        ----------
        rhs: FiniteFieldPolynomial
            The right-hand-side operand.
        var: Expression
            The variable with respect to which the resultant is computed.
        """

    def factor_square_free(self) -> list[tuple[FiniteFieldPolynomial, int]]:
        """
        Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial().to_finite_field(7)
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[tuple[FiniteFieldPolynomial, int]]:
        """
        Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial().to_finite_field(7)
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> FiniteFieldPolynomial:
        """
        Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.derivative(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to differentiate.
        """

    def integrate(self, x: Expression) -> FiniteFieldPolynomial:
        """
        Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.integrate(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to integrate.
        """

    def monic(self) -> FiniteFieldPolynomial:
        """
        Get the monic part of the polynomial, i.e., the polynomial
        divided by its leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.monic())  # x^2+1/2*x+3/2
        """

    def lcoeff(self) -> FiniteFieldPolynomial:
        """
        Get the leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
        >>> print(p)  # 3
        """

    def coefficient_list(self, xs: Expression | Sequence[Expression] | None = None) -> list[tuple[list[int], FiniteFieldPolynomial]]:
        """
        Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)

        Parameters
        ----------
        xs: Expression | Sequence[Expression] | None
            The variables with respect to which coefficients should be listed.
        """

    @classmethod
    def groebner_basis(_cls, system: list[FiniteFieldPolynomial], grevlex: bool = True, print_stats: bool = False) -> list[FiniteFieldPolynomial]:
        """
        Compute the Groebner basis of a polynomial system.

        Examples
        --------
        >>> basis = Polynomial.groebner_basis(
        >>>     [E("a b c d - 1").to_polynomial(),
        >>>     E("a b c + a b d + a c d + b c d").to_polynomial(),
        >>>     E("a b + b c + a d + c d").to_polynomial(),
        >>>     E("a + b + c + d").to_polynomial()],
        >>>     grevlex=True,
        >>>     print_stats=True
        >>> )
        >>> for p in basis:
        >>>     print(p)

        Parameters
        ----------
        grevlex: bool
            If `True`, reverse graded lexicographical ordering is used, otherwise the ordering is lexicographical.
        print_stats: bool
            If `True`, intermediate statistics will be printed.
        """

    def reduce(self, gs: Sequence[Polynomial], grevlex: bool = True) -> FiniteFieldPolynomial:
        """
        Completely reduce the polynomial w.r.t the polynomials `gs`.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        Examples
        --------
        >>> E('y^2+x').to_polynomial().reduce([E('x').to_polynomial()])

        yields `y^2`

        Parameters
        ----------
        gs: Sequence[Polynomial]
            The polynomials that define the reducing set.
        grevlex: bool
            Whether graded reverse lexicographic ordering should be used.
        """

    def evaluate(self, input: Sequence[int]) -> int:
        """
        Evaluate the polynomial at point `input`.

        Examples
        --------

        >>> from symbolica import *
        >>> P('x*y+2*x+x^2', modulus=5).evaluate([2, 3])

        Yields `4`.

        Parameters
        ----------
        input: Sequence[int]
            The input value.
        """

    def replace(self, x: Expression, v: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """
        Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> r = E('y+1').to_polynomial())
        >>> p.replace(S('x'), r)

        Parameters
        ----------
        x: Expression
            The variable to replace.
        v: FiniteFieldPolynomial | int
            The polynomial or scalar value that should replace `x`.
        """

    def to_expression(self) -> Expression:
        """
        Convert the polynomial to an expression.
        """

    def to_polynomial(self) -> Polynomial:
        """
        Convert a Galois field polynomial to a simple finite field polynomial.
        """

    def to_galois_field(self, min_poly: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """
        Convert the coefficients of the polynomial to a Galois field defined by the minimal polynomial `min_poly`.

        Parameters
        ----------
        min_poly: FiniteFieldPolynomial
            The minimal polynomial that defines the algebraic extension.
        """

    def get_minimal_polynomial(self) -> FiniteFieldPolynomial:
        """
        Get the minimal polynomial of the algebraic extension.
        """

    def get_modulus(self) -> int:
        """
        Get the modulus of the finite field.
        """

    def adjoin(self, b: FiniteFieldPolynomial, new_symbol: Expression | None = None) -> tuple[FiniteFieldPolynomial, FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """
        Adjoin the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
        is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.

        `b`  must be irreducible over `R` and `R[a]`; this is not checked.

        If `new_symbol` is provided, the variable of the new extension will be renamed to it.
        Otherwise, the variable of the new extension will be the same as that of `b`.

        Parameters
        ----------
        b: FiniteFieldPolynomial
            The finite-field polynomial that defines the extension to adjoin.
        new_symbol: Expression | None
            The symbol chosen for the adjoined generator.
        """

    def simplify_algebraic_number(self, min_poly: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """
        Find the minimal polynomial for the algebraic number represented by this polynomial
        expressed in the number field defined by `minimal_poly`.

        Parameters
        ----------
        min_poly: FiniteFieldPolynomial
            The minimal polynomial that defines the algebraic extension.
        """


class RationalPolynomial:
    """A Symbolica rational polynomial."""

    def __new__(_cls, num: Polynomial, den: Polynomial) -> RationalPolynomial:
        """
        Create a new rational polynomial from a numerator and denominator polynomial.

        Parameters
        ----------
        num: Polynomial
            The numerator polynomial.
        den: Polynomial
            The denominator polynomial.
        """

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], default_namespace: str | None = None) -> RationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = RationalPolynomial.parse('(3/4*x^2+y+y*4)/(1+x)', ['x', 'y'])

        Parameters
        ----------
        input: str
            The input value.
        vars: Sequence[str]
            The variables treated as polynomial variables, in the given order.
        default_namespace: str | None
            The namespace assumed for unqualified symbols during parsing.

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __copy__(self) -> RationalPolynomial:
        """
        Copy the rational polynomial.
        """

    def __str__(self) -> str:
        """
        Print the rational polynomial in a human-readable format.
        """

    def to_latex(self) -> str:
        """
        Convert the rational polynomial into a LaTeX string.
        """

    def get_variables(self) -> Sequence[Expression]:
        """
        Get the list of variables in the internal ordering of the polynomial.
        """

    def numerator(self) -> Polynomial:
        """
        Get the numerator.
        """

    def denominator(self) -> Polynomial:
        """
        Get the denominator.
        """

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two rational polynomials are equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two rational polynomials are not equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __lt__(self, rhs: int) -> bool:
        """
        Check if the rational polynomial is less than an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __le__(self, rhs: int) -> bool:
        """
        Check if the rational polynomial is less than or equal to an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __gt__(self, rhs: int) -> bool:
        """
        Check if the rational polynomial is greater than an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __ge__(self, rhs: int) -> bool:
        """
        Check if the polynomial is greater than or equal to an integer.

        Parameters
        ----------
        rhs: int
            The right-hand-side operand.
        """

    def __add__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """
        Add two rational polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: RationalPolynomial
            The right-hand-side operand.
        """

    def __sub__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """
        Subtract rational polynomials `rhs` from `self`, returning the result.

        Parameters
        ----------
        rhs: RationalPolynomial
            The right-hand-side operand.
        """

    def __mul__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """
        Multiply two rational polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: RationalPolynomial
            The right-hand-side operand.
        """

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """
        Divide the polynomial `self` by `rhs`, rounding down, returning the result.

        Parameters
        ----------
        rhs: Polynomial
            The right-hand-side operand.
        """

    def __truediv__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """
        Divide the rational polynomial `self` by `rhs` if possible, returning the result.

        Parameters
        ----------
        rhs: RationalPolynomial
            The right-hand-side operand.
        """

    def __neg__(self) -> RationalPolynomial:
        """
        Negate the rational polynomial.
        """

    def gcd(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """
        Compute the greatest common divisor (GCD) of two rational polynomials.

        Parameters
        ----------
        rhs: RationalPolynomial
            The right-hand-side operand.
        """

    def to_finite_field(self, prime: int) -> FiniteFieldRationalPolynomial:
        """
        Convert the coefficients of the rational polynomial to a finite field with prime `prime`.

        Parameters
        ----------
        prime: int
            The prime modulus of the target finite field.
        """

    def to_expression(self) -> Expression:
        """
        Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('(x*y+2*x+x^2)/(x^7+y+1)')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def derivative(self, x: Expression) -> RationalPolynomial:
        """
        Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> print(p.derivative(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to differentiate.
        """

    def apart(self, x: Expression | None = None) -> list[RationalPolynomial]:
        """
        Compute the partial fraction decomposition in `x`.

        If `None` is passed, the expression will be decomposed in all variables
        which involves a potentially expensive Groebner basis computation.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)

        Parameters
        ----------
        x: Expression | None
            The variable with respect to which to perform the partial-fraction decomposition.
        """


class FiniteFieldRationalPolynomial:
    """A Symbolica rational polynomial."""

    def __new__(_cls, num: FiniteFieldPolynomial, den: FiniteFieldPolynomial) -> FiniteFieldRationalPolynomial:
        """
        Create a new rational polynomial from a numerator and denominator polynomial.

        Parameters
        ----------
        num: FiniteFieldPolynomial
            The numerator polynomial.
        den: FiniteFieldPolynomial
            The denominator polynomial.
        """

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], prime: int, default_namespace: str | None = None) -> FiniteFieldRationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = FiniteFieldRationalPolynomial.parse('3*x^2+y+y*4', ['x', 'y'], 17)

        Parameters
        ----------
        input: str
            The input value.
        vars: Sequence[str]
            The variables treated as polynomial variables, in the given order.
        prime: int
            The prime modulus of the finite field.
        default_namespace: str | None
            The namespace assumed for unqualified symbols during parsing.

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """
        Check if two polynomials are not equal.

        Parameters
        ----------
        rhs: Polynomial | int
            The right-hand-side operand.
        """

    def __copy__(self) -> FiniteFieldRationalPolynomial:
        """
        Copy the rational polynomial.
        """

    def __str__(self) -> str:
        """
        Print the rational polynomial in a human-readable format.
        """

    def to_latex(self) -> str:
        """
        Convert the rational polynomial into a LaTeX string.
        """

    def get_variables(self) -> Sequence[Expression]:
        """
        Get the list of variables in the internal ordering of the polynomial.
        """

    def __add__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """
        Add two rational polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldRationalPolynomial
            The right-hand-side operand.
        """

    def __sub__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """
        Subtract rational polynomials `rhs` from `self`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldRationalPolynomial
            The right-hand-side operand.
        """

    def __mul__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """
        Multiply two rational polynomials `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldRationalPolynomial
            The right-hand-side operand.
        """

    def __truediv__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """
        Divide the rational polynomial `self` by `rhs` if possible, returning the result.

        Parameters
        ----------
        rhs: FiniteFieldRationalPolynomial
            The right-hand-side operand.
        """

    def __neg__(self) -> FiniteFieldRationalPolynomial:
        """
        Negate the rational polynomial.
        """

    def gcd(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """
        Compute the greatest common divisor (GCD) of two rational polynomials.

        Parameters
        ----------
        rhs: FiniteFieldRationalPolynomial
            The right-hand-side operand.
        """

    def get_modulus(self) -> int:
        """
        Get the modulus of the finite field.
        """

    def derivative(self, x: Expression) -> RationalPolynomial:
        """
        Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> print(p.derivative(x))

        Parameters
        ----------
        x: Expression
            The variable with respect to which to differentiate.
        """

    def apart(self, x: Expression | None = None) -> list[FiniteFieldRationalPolynomial]:
        """
        Compute the partial fraction decomposition in `x`.

        If `None` is passed, the expression will be decomposed in all variables
        which involves a potentially expensive Groebner basis computation.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)

        Parameters
        ----------
        x: Expression | None
            The variable with respect to which to perform the partial-fraction decomposition.
        """


class Matrix:
    """A matrix with rational polynomial coefficients."""

    def __new__(cls, nrows: int, ncols: int) -> Matrix:
        """
        Create a new zeroed matrix with `nrows` rows and `ncols` columns.

        Parameters
        ----------
        nrows: int
            The number of rows.
        ncols: int
            The number of columns.
        """

    @classmethod
    def identity(cls, nrows: int) -> Matrix:
        """
        Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere.

        Parameters
        ----------
        nrows: int
            The number of rows.
        """

    @classmethod
    def eye(cls, diag: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """
        Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere.

        Parameters
        ----------
        diag: Sequence[RationalPolynomial | Polynomial | Expression | int]
            The entries to place on the diagonal.
        """

    @classmethod
    def vec(cls, entries: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """
        Create a new column vector from a list of scalars.

        Parameters
        ----------
        entries: Sequence[RationalPolynomial | Polynomial | Expression | int]
            The entries of the column vector, from top to bottom.
        """

    @classmethod
    def from_linear(cls, nrows: int, ncols: int, entries: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """
        Create a new matrix from a 1-dimensional vector of scalars.

        Parameters
        ----------
        nrows: int
            The number of rows.
        ncols: int
            The number of columns.
        entries: Sequence[RationalPolynomial | Polynomial | Expression | int]
            The matrix entries in row-major order.
        """

    @classmethod
    def from_nested(cls, entries: Sequence[Sequence[RationalPolynomial | Polynomial | Expression | int]]) -> Matrix:
        """
        Create a new matrix from a 2-dimensional vector of scalars.

        Parameters
        ----------
        entries: Sequence[Sequence[RationalPolynomial | Polynomial | Expression | int]]
            The nested row entries of the matrix.
        """

    def nrows(self) -> int:
        """
        Get the number of rows in the matrix.
        """

    def ncols(self) -> int:
        """
        Get the number of columns in the matrix.
        """

    def is_zero(self) -> bool:
        """
        Return true iff every entry in the matrix is zero.
        """

    def is_diagonal(self) -> bool:
        """
        Return true iff every non- main diagonal entry in the matrix is zero.
        """

    def transpose(self) -> Matrix:
        """
        Return the transpose of the matrix.
        """

    def swap_rows(self, i: int, j: int, start: int = 0) -> None:
        """
        Swap rows `i` and `j` of the matrix in-place, starting from column `start`.

        Parameters
        ----------
        i: int
            The first index.
        j: int
            The second index.
        start: int
            The starting index or value.
        """

    def swap_cols(self, i: int, j: int) -> None:
        """
        Swap columns `i` and `j` of the matrix in-place.

        Parameters
        ----------
        i: int
            The first index.
        j: int
            The second index.
        """

    def inv(self) -> Matrix:
        """
        Return the inverse of the matrix, if it exists.
        """

    def det(self) -> RationalPolynomial:
        """
        Return the determinant of the matrix.
        """

    def solve(self, b: Matrix) -> Matrix:
        """
        Solve `A * x = b` for `x`, where `A` is the current matrix.

        Parameters
        ----------
        b: Matrix
            The right-hand-side matrix `b` in `A * x = b`.
        """

    def solve_any(self, b: Matrix) -> Matrix:
        """
        Solve `A * x = b` for `x`, where `A` is the current matrix and return any solution if the
        system is underdetermined.

        Parameters
        ----------
        b: Matrix
            The right-hand-side matrix `b` in `A * x = b`.
        """

    def row_reduce(self, max_col: int) -> int:
        """
        Row-reduce the first `max_col` columns of the matrix in-place using Gaussian elimination and return the rank.

        Parameters
        ----------
        max_col: int
            The highest column index included in row reduction.
        """

    def augment(self, b: Matrix) -> Matrix:
        """
        Augment the matrix with another matrix, e.g. create `[A B]` from matrix `A` and `B`.

        Returns an error when the matrices do not have the same number of rows.

        Parameters
        ----------
        b: Matrix
            The matrix to append as additional columns.
        """

    def split_col(self, col: int) -> tuple[Matrix, Matrix]:
        """
        Split the matrix into two matrices at column `col`.

        Parameters
        ----------
        col: int
            The column index at which to split the matrix.
        """

    def content(self) -> RationalPolynomial:
        """
        Get the content, i.e., the GCD of the coefficients.
        """

    def primitive_part(self) -> Matrix:
        """
        Construct the same matrix, but with the content removed.
        """

    def map(self, f: Callable[[RationalPolynomial], RationalPolynomial]) -> Matrix:
        """
        Apply a function `f` to every entry of the matrix.

        Parameters
        ----------
        f: Callable[[RationalPolynomial], RationalPolynomial]
            The callback or function to apply.
        """

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        max_line_length: int | None = 80,
        indentation: int = 4,
        fill_indented_lines: bool = True,
        pretty_matrix=True,
        number_thousands_separator: str | None = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        function_brackets: tuple[str, str] = ('(', ')'),
        num_exp_as_superscript: bool = True,
        precision: int | None = None,
        show_namespaces: bool = False,
        hide_namespace: str | None = None,
        include_attributes: bool = False,
        max_terms: int | None = None,
        custom_print_mode: int | None = None,
    ) -> str:
        """
        Convert the matrix into a human-readable string, with tunable settings.

        Parameters
        ----------
        mode: PrintMode
            The mode that controls how the input is interpreted or formatted.
        max_line_length: int | None
            The preferred maximum line length before wrapping.
        indentation: int
            The number of spaces used for wrapped lines.
        fill_indented_lines: bool
            Whether wrapped lines should be padded to the configured indentation.
        pretty_matrix: Any
            Whether matrices should be printed in the pretty multi-line layout.
        number_thousands_separator: str | None
            The separator inserted between groups of digits in printed integers.
        multiplication_operator: str
            The string used to print multiplication.
        double_star_for_exponentiation: bool
            Whether exponentiation should be printed as `**` instead of `^`.
        square_brackets_for_function: bool
            Whether function calls should be printed with square brackets.
        function_brackets: tuple[str, str]
            The opening and closing brackets used when printing function arguments.
        num_exp_as_superscript: bool
            Whether small integer exponents should be printed as superscripts.
        precision: int | None
            The decimal precision used when printing numeric coefficients.
        show_namespaces: bool
            Whether namespaces should be included in the formatted output.
        hide_namespace: str | None
            A namespace prefix to omit from printed symbol names.
        include_attributes: bool
            Whether symbol attributes should be included in the printed output.
        max_terms: int | None
            The maximum number of terms to print before truncating the output.
        custom_print_mode: int | None
            A custom print-mode identifier passed through to custom print callbacks.
        """

    def to_latex(self) -> str:
        """
        Convert the matrix into a LaTeX string.
        """

    def __copy__(self) -> Matrix:
        """
        Copy the matrix.
        """

    def __getitem__(self, key: tuple[int, int]) -> RationalPolynomial:
        """
        Get the entry at position `key` in the matrix.

        Parameters
        ----------
        key: tuple[int, int]
            The `(row, column)` index of the entry to retrieve.
        """

    def __str__(self) -> str:
        """
        Print the matrix in a human-readable format.
        """

    def __eq__(self, other: Matrix) -> bool:
        """
        Compare two matrices.

        Parameters
        ----------
        other: Matrix
            The other operand to combine or compare with.
        """

    def __neq__(self, other: Matrix) -> bool:
        """
        Compare two matrices.

        Parameters
        ----------
        other: Matrix
            The other operand to combine or compare with.
        """

    def __add__(self, rhs: Matrix) -> Matrix:
        """
        Add two matrices `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Matrix
            The right-hand-side operand.
        """

    def __sub__(self, rhs: Matrix) -> Matrix:
        """
        Subtract matrix `rhs` from `self`, returning the result.

        Parameters
        ----------
        rhs: Matrix
            The right-hand-side operand.
        """

    def __mul__(self, rhs: Matrix | RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """
        Matrix multiply `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Matrix | RationalPolynomial | Polynomial | Expression | int
            The right-hand-side operand.
        """

    def __rmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """
        Matrix multiply  `rhs` and `self`, returning the result.

        Parameters
        ----------
        rhs: RationalPolynomial | Polynomial | Expression | int
            The right-hand-side operand.
        """

    def __matmul__(self, rhs: Matrix | RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """
        Matrix multiply `self` and `rhs`, returning the result.

        Parameters
        ----------
        rhs: Matrix | RationalPolynomial | Polynomial | Expression | int
            The right-hand-side operand.
        """

    def __rmatmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """
        Matrix multiply  `rhs` and `self`, returning the result.

        Parameters
        ----------
        rhs: RationalPolynomial | Polynomial | Expression | int
            The right-hand-side operand.
        """

    def __truediv__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """
        Divide this matrix by scalar `rhs` and return the result.

        Parameters
        ----------
        rhs: RationalPolynomial | Polynomial | Expression | int
            The right-hand-side operand.
        """

    def __neg__(self) -> Matrix:
        """
        Negate the matrix, returning the result.
        """


class Evaluator:
    """An optimized evaluator of an expression."""

    def __copy__(self) -> Evaluator:
        """
        Copy the evaluator.
        """

    @classmethod
    def load(cls, evaluator: bytes, external_functions: dict[tuple[Expression, str], Callable[[
            Sequence[float | complex]], float | complex]] = {}) -> Evaluator:
        """
        Load the evaluator into memory, preparing it for evaluation.

        Parameters
        ----------
        evaluator: bytes
            The serialized evaluator state.
        external_functions: dict[tuple[Expression, str], Callable[[ Sequence[float | complex]], float | complex]]
            The external functions to register.
        """

    def jit_compile(self, jit_compile: bool) -> None:
        """
        JIT compile the evaluator for faster evaluation. This may take some time, but will speed up subsequent evaluations.

        Parameters
        ----------
        jit_compile: bool
            Whether JIT compilation should be enabled.
        """

    def save(self) -> bytes:
        """
        Save the evaluator to a byte string.
        """

    def get_instructions(self) -> tuple[list[tuple[str, tuple[str, int], list[tuple[str, int]]]], int, list[Expression]]:
        """
        Return the instructions for efficiently evaluating the expression, the length of the list
        of temporary variables, and the list of constants. This can be used to generate
        code for the expression evaluation in any programming language.

        There are four lists that are used in the evaluation instructions:
        - `param`: the list of input parameters.
        - `temp`: the list of temporary slots. The size of it is provided as the second return value.
        - `const`: the list of constants.
        - `out`: the list of outputs.

        The instructions are of the form:
        - `('add', ('out', 0), [('const', 1), ('param', 0)], 0)` which means `out[0] = const[1] + param[0]` where the first `0` arguments are real.
        - `('mul', ('out', 0), [('temp', 0), ('param', 0)], 1)` which means `out[0] = temp[0] * param[0]`, where the first `1` arguments are real.
        - `('pow', ('out', 0), ('param', 0), -1, true)` which means `out[0] = param[0]^-1` and the output is real (`true`).
        - `('powf', ('out', 0), ('param', 0), ('param', 1), false)` which means `out[0] = param[0]^param[1]`.
        - `('fun', ('temp', 1), f, ["0"], [('param', 0)], true)` which means `temp[1] = f(0, param[0])` and the output is real (`true`).
        - `('if_else', ('temp', 0), 5)` which means `if temp[0] == 0 goto label 5` (false branch).
        - `('goto', 10)` which means `goto label 10`.
        - `('label', 3)` which means `label 3`.
        - `('join', ('out', 0), ('temp', 0), 3, 7)` which means `out[0] = (temp[0] != 0) ? label 3 : label 7`.

        Examples
        --------

        >>> from symbolica import *
        >>> (ins, m, c) = E('x^2+5/3+cos(x)').evaluator({}, {}, [S('x')]).get_instructions()
        >>>
        >>> for x in ins:
        >>>     print(x)
        >>> print('temp list length:', m)
        >>> print('constants:', c)

        yields

        ```
        ('mul', ('out', 0), [('param', 0), ('param', 0)], 0)
        ('fun', ('temp', 1), cos, ('param', 0), false)
        ('add', ('out', 0), [('const', 0), ('out', 0), ('temp', 1)])
        temp list length: 2
        constants: [5/3]
        ```
        """

    def merge(self, other: Evaluator, cpe_iterations: int | None = None) -> None:
        """
        Merge evaluator `other` into `self`. The parameters must be the same, and
        the outputs will be concatenated.

        The optional `cpe_iterations` parameter can be used to limit the number of common
        pair elimination rounds after the merge.

        Examples
        --------

        >>> from symbolica import *
        >>> e1 = E('x').evaluator({}, {}, [S('x')])
        >>> e2 = E('x+1').evaluator({}, {}, [S('x')])
        >>> e1.merge(e2)
        >>> e1.evaluate([[2.]])

        yields `[2, 3]`.

        Parameters
        ----------
        other: Evaluator
            The other operand to combine or compare with.
        cpe_iterations: int | None
            The number of common subexpression elimination iterations to perform.
        """

    def dualize(self, dual_shape: list[list[int]], external_functions: dict[tuple[str, str, int], Callable[[
            Sequence[float | complex]], float | complex]] | None = None,
            zero_components: list[tuple[int, int]] | None = None) -> None:
        """
        Dualize the evaluator to support hyper-dual numbers with the given shape,
        indicating the number of derivatives in every variable per term.
        This allows for efficient computation of derivatives.

        For example, to compute first derivatives in two variables `x` and `y`,
        use `dual_shape = [[0, 0], [1, 0], [0, 1]]`.

        External functions must be mapped to `len(dual_shape)` different functions
        that compute a single component each. The input to the functions
        is the flattened vector of all components of all parameters,
        followed by all previously computed output components.

        Examples
        --------

        >>> from symbolica import *
        >>> e1 = E('x^2 + y*x').evaluator({}, {}, [S('x'), S('y')])
        >>> e1.dualize([[0, 0], [1, 0], [0, 1]])
        >>> r = e1.evaluate([[2., 1., 0., 3., 0., 1.]])
        >>> print(r)  # [10, 7, 2]

        Mapping external functions:

        >>> ev = E('f(x + 1)').evaluator({}, {}, [S('x')], external_functions={(S('f'), 'f'): lambda args: args[0]})
        >>> ev.dualize([[0], [1]], {('f', 'f0', 0): lambda args: args[0], ('f', 'f1', 1): lambda args: args[1]})
        >>> print(ev.evaluate([[2., 1.]]))  # [[3. 1.]]

        Parameters
        ----------
        dual_shape : list[list[int]]
            The shape of the dual numbers, indicating the number of derivatives
            in every variable per term.
        external_functions : dict[tuple[str, str, int], Callable[[Sequence[float | complex]], float | complex]] | None
            A mapping from external function identifiers to functions that compute a single component each.
            The key is a tuple of function name, unique printable name, and component index.
            The value is a function that takes the flattened parameters and returns a component.
        zero_components : list[tuple[int, int]] | None
            A list of components that are known to be zero and can be skipped in the dualization.
            Each component is specified as a tuple of (parameter index, dual index).
        """

    def set_real_params(self, real_params: list[int], sqrt_real=False, log_real=False, powf_real=False, verbose=False) -> None:
        """
        Set which parameters are fully real. This allows for more optimal
        assembly output that uses real arithmetic instead of complex arithmetic
        where possible.

        You can also set if all encountered sqrt, log, and powf operations with real
        arguments are expected to yield real results.

        Must be called after all optimization functions and merging are performed
        on the evaluator, or the registration will be lost.

        Parameters
        ----------
        real_params: list[int]
            The parameter indices that should be treated as real.
        sqrt_real: Any
            Whether square roots should be assumed real.
        log_real: Any
            Whether logarithms should be assumed real.
        powf_real: Any
            Whether fractional powers should be assumed real.
        verbose: Any
            Whether verbose output should be enabled.
        """

    @overload
    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        number_type: Literal['real'],
        inline_asm: str = 'default',
        optimization_level: int = 3,
        native: bool = True,
        compiler_path: str | None = None,
        compiler_flags: Sequence[str] | None = None,
        custom_header: str | None = None,
    ) -> CompiledRealEvaluator:
        """
        Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

        Parameters
        ----------
        function_name : str
            The name of the function to generate and compile.
        filename : str
            The name of the file to generate.
        library_name : str
            The name of the shared library to generate.
        number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
            The numeric backend to generate. Use 'real' for double precision or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The compiler optimization level. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : str | None
            The custom path to the compiler executable.
        compiler_flags : Sequence[str] | None
            The custom flags to pass to the compiler.
        custom_header : str | None
            The custom header to include in the generated code.
        """

    @overload
    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        number_type: Literal['complex'],
        inline_asm: str = 'default',
        optimization_level: int = 3,
        native: bool = True,
        compiler_path: str | None = None,
        compiler_flags: Sequence[str] | None = None,
        custom_header: str | None = None,
    ) -> CompiledComplexEvaluator:
        """
        Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

        Parameters
        ----------
        function_name : str
            The name of the function to generate and compile.
        filename : str
            The name of the file to generate.
        library_name : str
            The name of the shared library to generate.
        number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
            The numeric backend to generate. Use 'real' for double precision or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The compiler optimization level. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : str | None
            The custom path to the compiler executable.
        compiler_flags : Sequence[str] | None
            The custom flags to pass to the compiler.
        custom_header : str | None
            The custom header to include in the generated code.
        """

    @overload
    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        number_type: Literal['real_4x'],
        inline_asm: str = 'default',
        optimization_level: int = 3,
        native: bool = True,
        compiler_path: str | None = None,
        compiler_flags: Sequence[str] | None = None,
        custom_header: str | None = None,
    ) -> CompiledSimdRealEvaluator:
        """
        Compile the evaluator to a shared library with 4x SIMD using C++ and optionally inline assembly and load it.

        Parameters
        ----------
        function_name : str
            The name of the function to generate and compile.
        filename : str
            The name of the file to generate.
        library_name : str
            The name of the shared library to generate.
        number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
            The numeric backend to generate. Use 'real' for double precision or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The compiler optimization level. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : str | None
            The custom path to the compiler executable.
        compiler_flags : Sequence[str] | None
            The custom flags to pass to the compiler.
        custom_header : str | None
            The custom header to include in the generated code.
        """

    @overload
    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        number_type: Literal['complex_4x'],
        inline_asm: str = 'default',
        optimization_level: int = 3,
        native: bool = True,
        compiler_path: str | None = None,
        compiler_flags: Sequence[str] | None = None,
        custom_header: str | None = None,
    ) -> CompiledSimdComplexEvaluator:
        """
        Compile the evaluator to a shared library with 4x SIMD using C++ and optionally inline assembly and load it.

        Parameters
        ----------
        function_name : str
            The name of the function to generate and compile.
        filename : str
            The name of the file to generate.
        library_name : str
            The name of the shared library to generate.
        number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
            The numeric backend to generate. Use 'real' for double precision or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The compiler optimization level. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : str | None
            The custom path to the compiler executable.
        compiler_flags : Sequence[str] | None
            The custom flags to pass to the compiler.
        custom_header : str | None
            The custom header to include in the generated code.
        """

    @overload
    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        number_type: Literal['cuda_real'],
        inline_asm: str = 'default',
        optimization_level: int = 3,
        native: bool = True,
        compiler_path: str | None = None,
        compiler_flags: Sequence[str] | None = None,
        custom_header: str | None = None,
        cuda_number_of_evaluations: int | None = None,
        cuda_block_size: int | None = 256
    ) -> CompiledCudaRealEvaluator:
        """
        Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

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
            The numeric backend to generate. Use 'real' for double precision or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The compiler optimization level. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : str | None
            The custom path to the compiler executable.
        compiler_flags : Sequence[str] | None
            The custom flags to pass to the compiler.
        custom_header : str | None
            The custom header to include in the generated code.
        cuda_number_of_evaluations: int | None
            The number of parallel evaluations to perform on the CUDA device. The input to evaluate must
            have the length `cuda_number_of_evaluations * arg_len`.
        cuda_block_size: int | None
            The block size for CUDA kernel launches.
        """

    @overload
    def compile(
        self,
        function_name: str,
        filename: str,
        library_name: str,
        number_type: Literal['cuda_complex'],
        inline_asm: str = 'default',
        optimization_level: int = 3,
        native: bool = True,
        compiler_path: str | None = None,
        compiler_flags: Sequence[str] | None = None,
        custom_header: str | None = None,
        cuda_number_of_evaluations: int | None = None,
        cuda_block_size: int | None = 256
    ) -> CompiledCudaComplexEvaluator:
        """
        Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

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
        number_type :  Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
            The numeric backend to generate. Use 'real' for double precision or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The compiler optimization level. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : str | None
            The custom path to the compiler executable.
        compiler_flags : Sequence[str] | None
            The custom flags to pass to the compiler.
        custom_header : str | None
            The custom header to include in the generated code.
        cuda_number_of_evaluations: int | None
            The number of parallel evaluations to perform on the CUDA device. The input to evaluate must
            have the length `cuda_number_of_evaluations * arg_len`.
        cuda_block_size: int | None
            The block size for CUDA kernel launches.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Evaluate the expression for multiple inputs and return the result.
        For best performance, use `numpy` arrays instead of lists.

        On the first call, the expression is JIT compiled using SymJIT.

        Examples
        --------
        Evaluate the function for three sets of inputs:

        >>> from symbolica import *
        >>> import numpy as np
        >>> ev = E('x * y + 2').evaluator({}, {}, [S('x'), S('y')])
        >>> print(ev.evaluate(np.array([1., 2., 3., 4., 5., 6.]).reshape((3, 2))))

        Yields`[[ 4.] [ 8.] [14.]]`

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """

    def evaluate_with_prec(
        self, inputs: Sequence[float | str | Decimal], decimal_digit_precision: int
    ) -> list[Decimal]:
        """
        Evaluate the expression for a single input. The precision of the input parameters is honored, and
        all constants are converted to a float with a decimal precision set by `decimal_digit_precision`.

        If `decimal_digit_precision` is set to 32, a much faster evaluation using double-float arithmetic is performed.

        Examples
        --------
        Evaluate the function for a single input with 50 digits of precision:

        >>> from symbolica import *
        >>> ev = E('x^2').evaluator({}, {}, [S('x')])
        >>> print(ev.evaluate_with_prec([Decimal('1.234567890121223456789981273238947212312338947923')], 50))

        Yields `1.524157875318369274550121833760353508310334033629`

        Parameters
        ----------
        inputs: Sequence[float | str | Decimal]
            The input values or batches to evaluate.
        decimal_digit_precision: int
            The decimal precision used for arbitrary-precision evaluation.
        """

    def evaluate_complex(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Evaluate the expression for multiple inputs and return the result.
        For best performance, use `numpy` arrays and `np.complex128` instead of lists and
        `complex`.

        On the first call, the expression is JIT compiled using SymJIT.

        Examples
        --------
        Evaluate the function for three sets of inputs:

        >>> from symbolica import *
        >>> import numpy as np
        >>> ev = E('x * y + 2').evaluator({}, {}, [S('x'), S('y')])
        >>> print(ev.evaluate(np.array([1.+2j, 2., 3., 4., 5., 6.]).reshape((3, 2))))

        Yields`[[ 4.+4.j] [14.+0.j] [32.+0.j]]`

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """

    def evaluate_complex_with_prec(
        self,
        inputs: Sequence[tuple[float | str | Decimal, float | str | Decimal]],
        decimal_digit_precision: int,
    ) -> list[tuple[Decimal]]:
        """
        Evaluate the expression for a single complex input, represented as a tuple of real and imaginary parts.
        The precision of the input parameters is honored, and all constants are converted to a float with a decimal precision set by `decimal_digit_precision`.

        If `decimal_digit_precision` is set to 32, a much faster evaluation using double-float arithmetic is performed.

        Examples
        --------
        Evaluate the function for a single input with 50 digits of precision:

        >>> from symbolica import *
        >>> ev = E('x^2').evaluator({}, {}, [S('x')])
        >>> print(ev.evaluate_complex_with_prec(
        >>>     [(Decimal('1.234567890121223456789981273238947212312338947923'), Decimal('3.434567890121223356789981273238947212312338947923'))], 50))

        Yields `[(Decimal('-10.27209871653338252296233957800668637617803672307'), Decimal('8.480414467170121512062583245527383392798704790330'))]`

        Parameters
        ----------
        inputs: Sequence[tuple[float | str | Decimal, float | str | Decimal]]
            The input values or batches to evaluate.
        decimal_digit_precision: int
            The decimal precision used for arbitrary-precision evaluation.
        """


class CompiledRealEvaluator:
    """A compiled evaluator of an expression."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
    ) -> CompiledRealEvaluator:
        """
        Load a compiled library, previously generated with `Evaluator.compile()`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        function_name: str
            The exported symbol name of the compiled entry point.
        input_len: int
            The number of scalar inputs expected by the compiled evaluator.
        output_len: int
            The number of scalar outputs produced by the compiled evaluator.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Evaluate the expression for multiple inputs and return the result.

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """


class CompiledComplexEvaluator:
    """A compiled evaluator of an expression."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
    ) -> CompiledComplexEvaluator:
        """
        Load a compiled library, previously generated with `Evaluator.compile()`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        function_name: str
            The exported symbol name of the compiled entry point.
        input_len: int
            The number of scalar inputs expected by the compiled evaluator.
        output_len: int
            The number of scalar outputs produced by the compiled evaluator.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Evaluate the expression for multiple inputs and return the result.

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """


class CompiledSimdRealEvaluator:
    """A compiled evaluator of an expression that packs 4 double using SIMD."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
    ) -> CompiledSimdRealEvaluator:
        """
        Load a compiled library, previously generated with `Evaluator.compile()`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        function_name: str
            The exported symbol name of the compiled entry point.
        input_len: int
            The number of scalar inputs expected by the compiled evaluator.
        output_len: int
            The number of scalar outputs produced by the compiled evaluator.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Evaluate the expression for multiple inputs and return the result.

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """


class CompiledSimdComplexEvaluator:
    """A compiled evaluator of an expression that packs 4 double using SIMD."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
    ) -> CompiledSimdComplexEvaluator:
        """
        Load a compiled library, previously generated with `Evaluator.compile()`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        function_name: str
            The exported symbol name of the compiled entry point.
        input_len: int
            The number of scalar inputs expected by the compiled evaluator.
        output_len: int
            The number of scalar outputs produced by the compiled evaluator.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Evaluate the expression for multiple inputs and return the result.

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """


class CompiledCudaRealEvaluator:
    """A compiled evaluator of an expression that uses CUDA for GPU acceleration."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
        cuda_number_of_evaluations: int,
        cuda_block_size: int | None = 256
    ) -> CompiledCudaRealEvaluator:
        """
        Load a compiled library, previously generated with `Evaluator.compile()`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        function_name: str
            The exported symbol name of the compiled entry point.
        input_len: int
            The number of scalar inputs expected by the compiled evaluator.
        output_len: int
            The number of scalar outputs produced by the compiled evaluator.
        cuda_number_of_evaluations: int
            The number of evaluations to batch per CUDA kernel launch.
        cuda_block_size: int | None
            The CUDA thread block size used by the compiled kernel.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Evaluate the expression for multiple inputs and return the result.

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """


class CompiledCudaComplexEvaluator:
    """A compiled evaluator of an expression that uses CUDA for GPU acceleration."""

    @classmethod
    def load(
        _cls,
        filename: str,
        function_name: str,
        input_len: int,
        output_len: int,
        cuda_number_of_evaluations: int,
        cuda_block_size: int | None = 256
    ) -> CompiledCudaComplexEvaluator:
        """
        Load a compiled library, previously generated with `Evaluator.compile()`.

        Parameters
        ----------
        filename: str
            The file path to load from or save to.
        function_name: str
            The exported symbol name of the compiled entry point.
        input_len: int
            The number of scalar inputs expected by the compiled evaluator.
        output_len: int
            The number of scalar outputs produced by the compiled evaluator.
        cuda_number_of_evaluations: int
            The number of evaluations to batch per CUDA kernel launch.
        cuda_block_size: int | None
            The CUDA thread block size used by the compiled kernel.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Evaluate the expression for multiple inputs and return the result.

        Parameters
        ----------
        inputs: npt.ArrayLike
            The input values or batches to evaluate.
        """


class NumericalIntegrator:
    """A numerical integrator for high-dimensional integrals."""

    def __copy__(self) -> NumericalIntegrator:
        """
        Copy the grid without any unprocessed samples.
        """

    @classmethod
    def continuous(
        _cls,
        n_dims: int,
        n_bins: int = 128,
        min_samples_for_update: int = 100,
        bin_number_evolution: Sequence[int] | None = None,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        """
        Create a new continuous grid for the numerical integrator.

        Parameters
        ----------
        n_dims: int
            The number of continuous integration dimensions.
        n_bins: int
            The number of bins per continuous dimension.
        min_samples_for_update: int
            The minimum number of samples to accumulate before updating the grid.
        bin_number_evolution: Sequence[int] | None
            An optional schedule that changes the number of bins during training.
        train_on_avg: bool
            Whether integrator training should use average sample values.
        """

    @classmethod
    def discrete(
        _cls,
        bins: Sequence[NumericalIntegrator | None],
        max_prob_ratio: float = 100.0,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        """
        Create a new discrete grid for the numerical integrator. Each
        bin can have a sub-grid.

        Examples
        --------
        >>> def integrand(samples: list[Sample]):
        >>>     res = []
        >>>     for sample in samples:
        >>>         if sample.d[0] == 0:
        >>>             res.append(sample.c[0]**2)
        >>>         else:
        >>>             res.append(sample.c[0]**1/2)
        >>>     return res
        >>>
        >>> integrator = NumericalIntegrator.discrete(
        >>>     [NumericalIntegrator.continuous(1), NumericalIntegrator.continuous(1)])
        >>> integrator.integrate(integrand, True, 10, 10000)

        Parameters
        ----------
        bins: Sequence[NumericalIntegrator | None]
            The optional subgrid assigned to each discrete bin.
        max_prob_ratio: float
            The maximum probability ratio allowed between bins.
        train_on_avg: bool
            Whether integrator training should use average sample values.
        """

    @classmethod
    def uniform(
        _cls,
        bins: Sequence[int],
        continuous_subgrid: NumericalIntegrator,
    ) -> NumericalIntegrator:
        """
        Create a new uniform layered grid for the numerical integrator.
        `len(bins)` specifies the number of discrete layers, and each entry in `bins` specifies the number of bins in that layer.
        Each discrete bin has equal probability.

        Examples
        --------
        >>> def integrand(samples: Sequence[Sample]) -> list[float]:
        >>>     res = []
        >>>     for sample in samples:
        >>>         if sample.d[0] == 0:
        >>>             res.append(sample.c[0]**2)
        >>>         else:
        >>>             res.append(sample.c[0]**3)
        >>>     return res
        >>>
        >>>
        >>> integrator = NumericalIntegrator.uniform(
        >>>     [2], NumericalIntegrator.continuous(1))
        >>> integrator.integrate(integrand, min_error=1e-3)

        Parameters
        ----------
        bins: Sequence[int]
            The number of bins in each discrete layer.
        continuous_subgrid: NumericalIntegrator
            The continuous subgrid attached beneath the discrete layers.
        """

    @classmethod
    def rng(
        _cls,
        seed: int,
        stream_id: int
    ) -> RandomNumberGenerator:
        """
        Create a new random number generator, suitable for use with the integrator.
        Each thread of instance of the integrator should have its own random number generator,
        that is initialized with the same seed but with a different stream id.

        Parameters
        ----------
        seed: int
            The seed used to initialize the random number generator.
        stream_id: int
            The stream identifier for the random number generator.
        """

    @classmethod
    def import_grid(
        _cls,
        grid: bytes
    ) -> NumericalIntegrator:
        """
        Import an exported grid from another thread or machine.
        Use `export_grid` to export the grid.

        Parameters
        ----------
        grid: bytes
            The serialized integration grid to import.
        """

    def export_grid(
        self,
        export_samples: bool = True,
    ) -> bytes:
        """
        Export the grid, so that it can be sent to another thread or machine.
        If you are exporting your main grid, make sure to set `export_samples` to `False` to avoid copying unprocessed samples.

        Use `import_grid` to load the grid.

        Parameters
        ----------
        export_samples: bool
            Whether pending samples should be included in the exported grid.
        """

    def get_live_estimate(
        self,
    ) -> tuple[float, float, float, float, float, int]:
        """
        Get the estamate of the average, error, chi-squared, maximum negative and positive evaluations, and the number of processed samples
        for the current iteration, including the points submitted in the current iteration.
        """

    def probe(self, probe: Probe) -> float:
        """
        Probe the Jacobian weight for a region in the grid.

        Parameters
        ----------
        probe: Probe
            The probe that identifies the region of interest.
        """

    def sample(self, num_samples: int, rng: RandomNumberGenerator) -> list[Sample]:
        """
        Sample `num_samples` points from the grid using the random number generator
        `rng`. See `rng()` for how to create a random number generator.

        Parameters
        ----------
        num_samples: int
            The number of samples to draw.
        rng: RandomNumberGenerator
            The random number generator used to draw the samples.
        """

    def merge(self, other: NumericalIntegrator) -> None:
        """
        Add the accumulated training samples from the grid `other` to the current grid.
        The grid structure of `self` and `other` must be equivalent.

        Parameters
        ----------
        other: NumericalIntegrator
            The other operand to combine or compare with.
        """

    def add_training_samples(self, samples: Sequence[Sample], evals: Sequence[float]) -> None:
        """
        Add the samples and their corresponding function evaluations to the grid.
        Call `update` after to update the grid and to obtain the new expected value for the integral.

        Parameters
        ----------
        samples: Sequence[Sample]
            The samples to add or process.
        evals: Sequence[float]
            The function evaluations associated with the samples.
        """

    def update(self, discrete_learning_rate: float, continous_learning_rate: float) -> tuple[float, float, float]:
        """
        Update the grid using the `discrete_learning_rate` and `continuous_learning_rate`.
        Examples
        --------
        >>> from symbolica import NumericalIntegrator, Sample
        >>>
        >>> def integrand(samples: list[Sample]):
        >>>     res = []
        >>>     for sample in samples:
        >>>         res.append(sample.c[0]**2+sample.c[1]**2)
        >>>     return res
        >>>
        >>> integrator = NumericalIntegrator.continuous(2)
        >>> for i in range(10):
        >>>     samples = integrator.sample(10000 + i * 1000)
        >>>     res = integrand(samples)
        >>>     integrator.add_training_samples(samples, res)
        >>>     avg, err, chi_sq = integrator.update(1.5, 1.5)
        >>>     print('Iteration {}: {:.6} +- {:.6}, chi={:.6}'.format(i+1, avg, err, chi_sq))

        Parameters
        ----------
        discrete_learning_rate: float
            The learning rate for discrete layers.
        continous_learning_rate: float
            The learning rate for continuous layers.
        """

    def integrate(
        self,
        integrand: Callable[[Sequence[Sample]], list[float]],
        max_n_iter: int = 10000000,
        min_error: float = 0.01,
        n_samples_per_iter: int = 10000,
        seed: int = 0,
        show_stats: bool = True,
    ) -> tuple[float, float, float]:
        """
        Integrate the function `integrand` that maps a list of `Sample`s to a list of `float`s.
        The return value is the average, the statistical error, and chi-squared of the integral.

        With `show_stats=True`, intermediate statistics will be printed. `max_n_iter` determines the number
        of iterations and `n_samples_per_iter` determine the number of samples per iteration. This is
        the same amount of samples that the integrand function will be called with.

        For more flexibility, use `sample`, `add_training_samples` and `update`. See `update` for an example.

        Examples
        --------
        >>> from symbolica import NumericalIntegrator, Sample
        >>>
        >>> def integrand(samples: list[Sample]):
        >>>     res = []
        >>>     for sample in samples:
        >>>         res.append(sample.c[0]**2+sample.c[1]**2)
        >>>     return res
        >>>
        >>> avg, err = NumericalIntegrator.continuous(2).integrate(integrand, True, 10, 100000)
        >>> print('Result: {} +- {}'.format(avg, err))

        Parameters
        ----------
        integrand: Callable[[Sequence[Sample]], list[float]]
            The function to integrate.
        max_n_iter: int
            The maximum number of integration iterations.
        min_error: float
            The target statistical error.
        n_samples_per_iter: int
            The number of samples drawn per integration iteration.
        seed: int
            The seed used to initialize the random number generator.
        show_stats: bool
            Whether intermediate integration statistics should be shown.
        """


class Sample:
    """A sample from the Symbolica integrator. It could consist of discrete layers,
    accessible with `d` (empty when there are not discrete layers), and the final continuous layer `c` if it is present."""

    """ The weights the integrator assigned to this sample point, given in descending order:
    first the discrete layer weights and then the continuous layer weight."""
    weights: list[float]
    d: list[int]
    """ A sample point per (nested) discrete layer. Empty if not present."""
    c: list[float]
    """ A sample in the continuous layer. Empty if not present."""


class Probe:
    """A probe that is used to access the Jacobian weight of a point or region
    of interest.

    For continuous probes, `None` skips that dimension and includes the full
    range of the dimension (Jacobian weight of 1).

    For discrete probes, the first vector specifies a path through nested
    discrete grids, and the second vector specifies the final continuous probe.
    The path may stop before the full grid depth, in which case the remaining
    sub-Jacobian weight is 1 and the continuous probe must be empty.

    For uniform probes, `None` in the discrete indices skips that discrete
    dimension and includes its full range (Jacobian weight of 1)."""

    d: list[int]
    """ A sample point per (nested) discrete layer. Empty if not present."""
    c: list[float | None]
    """ A sample in the continuous layer. Empty if not present."""
    u: list[int | None]
    """ A sample in the uniform layer. Empty if not present."""

    @classmethod
    def discrete(_cls, d: list[int], c: list[float | None] | None = None) -> Probe:
        """
        Create a probe with the given discrete indices and optional continuous sample.
        The discrete indices are allowed to be less deep than the grid depth.

        Parameters
        ----------
        d: list[int]
            The discrete probe indices for each nested discrete layer.
        c: list[float | None] | None
            The continuous probe coordinates; use `None` to include the full range of a dimension.
        """

    @classmethod
    def continuous(_cls, c: list[float | None]) -> Probe:
        """
        Create a probe with the given continuous sample. Entering `None` skips that dimension and includes the full
        range of the dimension (Jacobian weight of 1).

        Parameters
        ----------
        c: list[float | None]
            The continuous probe coordinates; use `None` to include the full range of a dimension.
        """

    @classmethod
    def uniform(
        _cls, u: list[int | None], c: list[float | None] | None = None
    ) -> Probe:
        """
        Create a probe with the given uniform indices and optional continuous sample.
        Entering `None` skips that dimension and includes the full
        range of the dimension (Jacobian weight of 1).

        Parameters
        ----------
        u: list[int | None]
            The uniform discrete probe indices; use `None` to include the full range of a layer.
        c: list[float | None] | None
            The continuous probe coordinates; use `None` to include the full range of a dimension.
        """


class RandomNumberGenerator:
    """A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
    A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).

    Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
    which is an instance counter starting at 0."""

    def __new__(_cls, seed: int, stream_id: int):
        """
        Create a new random number generator with a given `seed` and `stream_id`. For parallel runs,
        each thread or instance generating samples should use the same `seed` but a different `stream_id`.

        Parameters
        ----------
        seed: int
            The seed used to initialize the random number generator.
        stream_id: int
            The stream identifier for the random number generator.
        """

    def __copy__(self) -> RandomNumberGenerator:
        """
        Copy the random number generator, so that the copy will generate the same sequence of random numbers.
        """

    def next(self) -> int:
        """
        Generate the next random unsigned 64-bit integer in the sequence.
        """

    def next_float(self) -> float:
        """
        Generate the next random floating-point number in the sequence, uniformly distributed in the range [0, 1).
        """

    @classmethod
    def load(_cls, state: bytes) -> RandomNumberGenerator:
        """
        Import a random number generator from a previously exported state. The state should be a bytes object of length 32.

        Parameters
        ----------
        state: bytes
            The serialized state to load.
        """

    def save(self) -> bytes:
        """
        Export the random number generator state as a bytes object of length 32, which can be imported again to restore the state.
        """


class HalfEdge:
    """A half-edge in a graph that connects to one vertex, consisting of a direction (or `None` if undirected) and edge data."""

    def __new__(_cls, data: Expression | int, direction: bool | None = None):
        """
        Create a new half-edge. The `data` can be any expression, and the `direction` can be `True` (outgoing),
        `False` (incoming) or `None` (undirected).

        Parameters
        ----------
        data: Expression | int
            The data to associate with the object.
        direction: bool | None
            The direction of the edge or half-edge.
        """

    def flip(self) -> HalfEdge:
        """
        Return a new half-edge with the direction flipped (if it has a direction).
        """

    def direction(self) -> bool | None:
        """
        Get the direction of the half-edge. `True` means outgoing, `False` means incoming, and `None` means undirected.
        """

    def data(self) -> Expression:
        """
        Get the data of the half-edge.
        """


class Graph:
    """A graph that supported directional edges, parallel edges, self-edges and expression data on the nodes and edges.

    Warning: modifying the graph if it is contained in a `dict` or `set` will invalidate the hash.
    """

    def __new__(_cls):
        """
        Create a new empty graph.
        """

    def __str__(self) -> str:
        """
        Print the graph in a human-readable format.
        """

    def __hash__(self) -> int:
        """
        Hash the graph.
        """

    def __copy__(self) -> Graph:
        """
        Copy the graph.
        """

    def __len__(self) -> int:
        """
        Get the number of nodes in the graph.
        """

    def __eq__(self, other: Graph) -> bool:
        """
        Compare two graphs.

        Parameters
        ----------
        other: Graph
            The other operand to combine or compare with.
        """

    def __neq__(self, other: Graph) -> bool:
        """
        Compare two graphs.

        Parameters
        ----------
        other: Graph
            The other operand to combine or compare with.
        """

    def __getitem__(self, idx: int) -> tuple[Sequence[int], Expression]:
        """
        Get the `idx`th node, consisting of the edge indices and the data.

        Parameters
        ----------
        idx: int
            The zero-based index to access.
        """

    @classmethod
    def generate(_cls,
                 external_nodes: Sequence[tuple[Expression | int, HalfEdge]],
                 vertex_signatures: Sequence[Sequence[HalfEdge]],
                 max_vertices: int | None = None,
                 max_loops: int | None = None,
                 max_bridges: int | None = None,
                 allow_self_loops: bool = False,
                 allow_zero_flow_edges: bool = False,
                 filter_fn: Callable[[Graph, int], bool] | None = None,
                 progress_fn: Callable[[Graph], bool] | None = None) -> dict[Graph, Expression]:
        """
        Generate all connected graphs with `external_edges` half-edges and the given allowed list
        of vertex connections. The vertex signatures are given in terms of an edge direction (or `None` if
        there is no direction) and edge data.

        Returns the canonical form of the graph and the size of its automorphism group (including edge permutations).
        If `KeyboardInterrupt` is triggered during the generation, the generation will stop and will yield the currently generated
        graphs.

        Examples
        --------
        >>> from symbolica import *
        >>> g, q = HalfEdge(S("g")), HalfEdge(S("q"), True)
        >>> graphs = Graph.generate(
        >>>     [(1, g), (2, g)],
        >>>     [[g, g, g], [g, g, g, g], [q.flip(), q, g]],
        >>>     max_loops=2,
        >>> )
        >>> for (g, sym) in graphs.items():
        >>>     print(f'Symmetry factor = 1/{sym}:')
        >>>     print(g.to_dot())

        generates all connected graphs up to 2 loops with the specified vertices.

        Parameters
        ----------
        external_nodes: Sequence[tuple[Expression | int, HalfEdge]]
            The external edges, consisting of a tuple of the node data and a tuple of the edge direction and edge data.
            If the node data is the same, flip symmetries will be recognized.
        vertex_signatures: Sequence[Sequence[HalfEdge]]
            The allowed connections for each vertex.
        max_vertices: int, optional
            The maximum number of vertices in the graph.
        max_loops: int, optional
            The maximum number of loops in the graph.
        max_bridges: int, optional
            The maximum number of bridges in the graph.
        allow_self_loops: bool, optional
            Whether self-edges are allowed.
        allow_zero_flow_edges: bool, optional
            Whether bridges that do not need to be crossed to connect external vertices are allowed.
        filter_fn: Callable[[Graph, int], bool] | None, optional
            Set a filter function that is called during the graph generation.
            The first argument is the graph `g` and the second argument the vertex count `n`
            that specifies that the first `n` vertices are completed (no new edges will) be
            assigned to them. The filter function should return `true` if the current
            incomplete graph is allowed, else it should return `false` and the graph is discarded.
        progress_fn: Callable[[Graph], bool] | None, optional
            Set a progress function that is called every time a new unique graph is created.
            The argument is the currently generated graph.
            If the function returns `false`, the generation is aborted and the currently
            generated graphs are returned.
        """

    def to_dot(self) -> str:
        """
        Convert the graph to a graphviz dot string.
        """

    def to_mermaid(self) -> str:
        """
        Convert the graph to a mermaid string.
        """

    def num_nodes(self) -> int:
        """
        Get the number of nodes in the graph.
        """

    def num_edges(self) -> int:
        """
        Get the number of edges in the graph.
        """

    def num_loops(self) -> int:
        """
        Get the number of loops in the graph.
        """

    def node(self, idx: int) -> tuple[Sequence[int], Expression]:
        """
        Get the `idx`th node, consisting of the edge indices and the data.

        Parameters
        ----------
        idx: int
            The zero-based index to access.
        """

    def nodes(self) -> list[tuple[Sequence[int], Expression]]:
        """
        Get all nodes, consisting of the edge indices and the data.
        """

    def edge(self, idx: int) -> tuple[int, int, bool, Expression]:
        """
        Get the `idx`th edge, consisting of the the source vertex, target vertex, whether the edge is directed, and the data.

        Parameters
        ----------
        idx: int
            The zero-based index to access.
        """

    def edges(self) -> list[tuple[int, int, bool, Expression]]:
        """
        Get all edges, consisting of the the source vertex, target vertex, whether the edge is directed, and the data.
        """

    def add_node(self, data: Expression | int | None = None) -> int:
        """
        Add a node with data `data` to the graph, returning the index of the node.
        The default data is the number 0.

        Parameters
        ----------
        data: Expression | int | None
            The data to associate with the object.
        """

    def add_edge(self, source: int, target: int, directed: bool = False, data: Expression | int | None = None) -> int:
        """
        Add an edge between the `source` and `target` nodes, returning the index of the edge.

        Optionally, the edge can be set as directed. The default data is the number 0.

        Parameters
        ----------
        source: int
            The source node index.
        target: int
            The target node index.
        directed: bool
            Whether the edge is directed.
        data: Expression | int | None
            The data to associate with the object.
        """

    def set_node_data(self, index: int, data: Expression | int) -> Expression:
        """
        Set the data of the node at index `index`, returning the old data.

        Parameters
        ----------
        index: int
            The index of the node whose data should be replaced.
        data: Expression | int
            The data to associate with the object.
        """

    def set_edge_data(self, index: int, data: Expression | int) -> Expression:
        """
        Set the data of the edge at index `index`, returning the old data.

        Parameters
        ----------
        index: int
            The index of the edge whose data should be replaced.
        data: Expression | int
            The data to associate with the object.
        """

    def set_directed(self, index: int, directed: bool) -> bool:
        """
        Set the directed status of the edge at index `index`, returning the old value.

        Parameters
        ----------
        index: int
            The index of the edge whose direction flag should be changed.
        directed: bool
            Whether the edge is directed.
        """

    def canonize(self) -> tuple[Graph, Sequence[int], Expression, Sequence[int]]:
        """
        Write the graph in a canonical form. Returns the canonized graph, the vertex map, the automorphism group size, and the orbit.
        """

    def canonize_edges(self) -> None:
        """
        Sort and relabel the edges of the graph, keeping the vertices fixed.
        """

    def is_isomorphic(self, other: Graph) -> bool:
        """
        Check if the graph is isomorphic to another graph.

        Parameters
        ----------
        other: Graph
            The other operand to combine or compare with.
        """


class Integer:
    @classmethod
    def prime_iter(_cls, start: int = 1) -> Iterator[int]:
        """
        Create an iterator over all 64-bit prime numbers starting from `start`.

        Parameters
        ----------
        start: int
            The starting index or value.
        """

    @classmethod
    def is_prime(_cls, n: int) -> bool:
        """
        Check if the 64-bit number `n` is a prime number.

        Parameters
        ----------
        n: int
            The integer to test for primality.
        """

    @classmethod
    def factor(_cls, n: int) -> Sequence[tuple[int, int]]:
        """
        Factor the 64-bit number `n` into its prime factors and return a list of tuples `(p, e)` where `p` is a prime factor and `e` is its exponent.

        Parameters
        ----------
        n: int
            The integer to factor.
        """

    @classmethod
    def totient(_cls, n: int) -> int:
        """
        Compute the Euler totient function of the number `n`, i.e., the number of integers less than `n` that are coprime to `n`.

        Parameters
        ----------
        n: int
            The integer whose Euler totient should be computed.
        """

    @classmethod
    def gcd(_cls, a: int, b: int) -> int:
        """
        Compute the greatest common divisor of the numbers `a` and `b`.

        Parameters
        ----------
        a: int
            The first integer.
        b: int
            The second integer.
        """

    @classmethod
    def lcm(_cls, a: int, b: int) -> int:
        """
        Compute the least common multiple of the numbers `a` and `b`.

        Parameters
        ----------
        a: int
            The first integer.
        b: int
            The second integer.
        """

    @classmethod
    def extended_gcd(_cls, a: int, b: int) -> tuple[int, int, int]:
        """
        Compute the greatest common divisor of the numbers `a` and `b` and the Bézout coefficients.

        Parameters
        ----------
        a: int
            The first integer.
        b: int
            The second integer.
        """

    @classmethod
    def chinese_remainder(_cls, n1: int, m1: int, n2: int, m2: int) -> int:
        """
        Solve the Chinese remainder theorem for the equations:
        `x = n1 mod m1` and `x = n2 mod m2`.

        Parameters
        ----------
        n1: int
            The first residue.
        m1: int
            The modulus for the first congruence.
        n2: int
            The second residue.
        m2: int
            The modulus for the second congruence.
        """

    @classmethod
    def solve_integer_relation(_cls, x: Sequence[int | float | complex | Decimal], tolerance: float | Decimal, max_coeff: int | None = None, gamma: float | Decimal | None = None) -> Sequence[int]:
        """
        Use the PSLQ algorithm to find a vector of integers `a` that satisfies `a.x = 0`,
        where every element of `a` is less than `max_coeff`, using a specified tolerance and number
        of iterations. The parameter `gamma` must be more than or equal to `2/sqrt(3)`.

        Examples
        --------
        Solve a `32.0177=b*pi+c*e` where `b` and `c` are integers:

        >>> r = Integer.solve_integer_relation([-32.0177, 3.1416, 2.7183], 1e-5, 100)
        >>> print(r)  # [1,5,6]
        Parameters
        ----------
        x: Sequence[int | float | complex | Decimal]
            The numeric vector for which an integer relation is sought.
        tolerance: float | Decimal
            The tolerance used to accept an integer relation.
        max_coeff: int | None
            The maximum coefficient size to consider.
        gamma: float | Decimal | None
            The PSLQ gamma parameter controlling the reduction strategy.
        """
