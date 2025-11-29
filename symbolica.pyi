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
from typing import Any, Callable, Literal, overload, Iterator, Optional, Sequence, Tuple, List
from decimal import Decimal
import numpy as np
import numpy.typing as npt


def use_custom_logger() -> None:
    """Enable logging using Python's logging module instead of using the default logging.
    This is useful when using Symbolica in a Jupyter notebook or other environments
    where stdout is not easily accessible.

    This function must be called before any Symbolica logging events are emitted.
    """


def get_version() -> str:
    """Get the current Symbolica version."""


def is_licensed() -> bool:
    """Check if the current Symbolica instance has a valid license key set."""


def set_license_key(key: str) -> None:
    """This function is deprecated and no longer sets the license key.
    You can set the license key through the environment variable `SYMBOLICA_LICENSE` before importing Symbolica:

    >>> import os  # noqa: E402
    >>> os.environ['SYMBOLICA_LICENSE'] = 'YOURKEY'  # noqa: E402
    >>> from symbolica import *
    """


def request_hobbyist_license(name: str, email: str) -> None:
    """Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address `email`."""


def request_trial_license(name: str, email: str, company: str) -> None:
    """Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address `email`."""


def request_sublicense(name: str, email: str, company: str, super_licence: str) -> None:
    """Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
    The key will be sent to the e-mail address `email`."""


def get_license_key(email: str) -> str:
    """Get the license key for the account registered with the provided email address."""


@overload
def S(name: str,
      is_symmetric: Optional[bool] = None,
      is_antisymmetric: Optional[bool] = None,
      is_cyclesymmetric: Optional[bool] = None,
      is_linear: Optional[bool] = None,
      is_scalar: Optional[bool] = None,
      is_real: Optional[bool] = None,
      is_integer: Optional[bool] = None,
      is_positive: Optional[bool] = None,
      tags: Optional[Sequence[str]] = None,
      custom_normalization: Optional[Transformer] = None,
      custom_print: Optional[Callable[..., Optional[str]]] = None,
      custom_derivative: Optional[Callable[[Expression, int], Expression]] = None) -> Expression:
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
    >>> print(e)
    x**2 + 5

    Define a regular symbol and use it as a function:
    >>> f = S('f')
    >>> e = f(1,2)
    >>> print(e)
    f(1,2)


    Define a symmetric function:
    >>> f = S('f', is_symmetric=True)
    >>> e = f(2,1)
    >>> print(e)
    f(1,2)


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

    Parameters
    ----------
    name : str
        The name of the symbol
    is_symmetric : Optional[bool]
        Set to true if the symbol is symmetric.
    is_antisymmetric : Optional[bool]
        Set to true if the symbol is antisymmetric.
    is_cyclesymmetric : Optional[bool]
        Set to true if the symbol is cyclesymmetric.
    is_linear : Optional[bool]
        Set to true if the symbol is linear.
    is_scalar : Optional[bool]
        Set to true if the symbol is a scalar. It will be moved out of linear functions.
    is_real : Optional[bool]
        Set to true if the symbol is a real number.
    is_integer : Optional[bool]
        Set to true if the symbol is an integer.
    is_positive : Optional[bool]
        Set to true if the symbol is a positive number.
    tags: Optional[Sequence[str]] = None
        A list of tags to associate with the symbol.
    custom_normalization : Optional[Transformer]
        A transformer that is called after every normalization. Note that the symbol
        name cannot be used in the transformer as this will lead to a definition of the
        symbol. Use a wildcard with the same attributes instead.
    custom_print : Optional[Callable[..., Optional[str]]]:
        A function that is called when printing the variable/function, which is provided as its first argument.
        This function should return a string, or `None` if the default print function should be used.
        The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
    custom_derivative: Optional[Callable[[Expression, int], Expression]]:
        A function that is called when computing the derivative of a function in a given argument.
    """


@overload
def S(*names: str,
      is_symmetric: Optional[bool] = None,
      is_antisymmetric: Optional[bool] = None,
      is_cyclesymmetric: Optional[bool] = None,
      is_linear: Optional[bool] = None,
      is_scalar: Optional[bool] = None,
      is_real: Optional[bool] = None,
      is_integer: Optional[bool] = None,
      is_positive: Optional[bool] = None,
      tags: Optional[Sequence[str]] = None) -> Sequence[Expression]:
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
    >>> print(e)
    f(1,2)

    Parameters
    ----------
    name : str
        The name of the symbol
    is_symmetric : Optional[bool]
        Set to true if the symbol is symmetric.
    is_antisymmetric : Optional[bool]
        Set to true if the symbol is antisymmetric.
    is_cyclesymmetric : Optional[bool]
        Set to true if the symbol is cyclesymmetric.
    is_linear : Optional[bool]
        Set to true if the symbol is multilinear.
    is_scalar : Optional[bool]
        Set to true if the symbol is a scalar. It will be moved out of linear functions.
    is_real : Optional[bool]
        Set to true if the symbol is a real number.
    is_integer : Optional[bool]
        Set to true if the symbol is an integer.
    is_positive : Optional[bool]
        Set to true if the symbol is a positive number.
    tags: Optional[Sequence[str]] = None
        A list of tags to associate with the symbol.
    """


def N(num: int | float | complex | str | Decimal, relative_error: Optional[float] = None) -> Expression:
    """Create a new Symbolica number from an int, a float, or a string.
    A floating point number is kept as a float with the same precision as the input,
    but it can also be converted to the smallest rational number given a `relative_error`.

    Examples
    --------
    >>> e = N(1) / 2
    >>> print(e)
    1/2

    >>> print(N(1/3))
    >>> print(N(0.33, 0.1))
    >>> print(N('0.333`3'))
    >>> print(N(Decimal('0.1234')))
    3.3333333333333331e-1
    1/3
    3.33e-1
    1.2340e-1
    """


def E(input: str, mode: ParseMode = ParseMode.Symbolica, default_namespace: str = "python") -> Expression:
    """
    Parse a Symbolica expression from a string.

    Parameters
    ----------
    input: str
        An input string. UTF-8 characters are allowed.
    mode: ParseMode
        The parsing mode to use. Use `ParseMode.Mathematica` to parse Mathematica expressions.
    default_namespace: str
        The default namespace to use when parsing symbols.

    Examples
    --------
    >>> e = E('x^2+y+y*4')
    >>> print(e)

    `x^2+5*y`

    >>> e = E('Cos[test`x] (2+ 3 I)', mode=ParseMode.Mathematica)
    >>> print(e)

    `cos(test::x)(2+3i)`

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
def P(poly: str, default_namespace: str = "python", vars: Optional[Sequence[Expression]] = None) -> Polynomial:
    """Parse a string a polynomial, optionally, with the variable ordering specified in `vars`.
    All non-polynomial parts will be converted to new, independent variables.
    """


@overload
def P(poly: str,  minimal_poly: Polynomial, default_namespace: str = "python", vars: Optional[Sequence[Expression]] = None,
      ) -> NumberFieldPolynomial:
    """Parse string to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    All non-polynomial elements will be converted to new independent variables.

    The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
    The minimal polynomial must be a monic, irreducible univariate polynomial.
    """


@overload
def P(poly: str,
      modulus: int,
      default_namespace: str = "python",
      power: Optional[Tuple[int, Expression]] = None,
      minimal_poly: Optional[Polynomial] = None,
      vars: Optional[Sequence[Expression]] = None,
      ) -> FiniteFieldPolynomial:
    """Parse a string to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    All non-polynomial elements will be converted to new independent variables.

    The coefficients will be converted to finite field elements modulo `modulus`.
    If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
    `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

    If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.
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
    head: Optional[str]
    """The string data of this atom."""
    tail: List[AtomTree]
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

    CONJ: Expression
    """The built-in complex conjugate function."""

    @overload
    @classmethod
    def symbol(_cls,
               name: str,
               is_symmetric: Optional[bool] = None,
               is_antisymmetric: Optional[bool] = None,
               is_cyclesymmetric: Optional[bool] = None,
               is_linear: Optional[bool] = None,
               is_scalar: Optional[bool] = None,
               is_real: Optional[bool] = None,
               is_integer: Optional[bool] = None,
               is_positive: Optional[bool] = None,
               tags: Optional[Sequence[str]] = None,
               custom_normalization: Optional[Transformer] = None,
               custom_print: Optional[Callable[..., Optional[str]]] = None,
               custom_derivative: Optional[Callable[[Expression, int], Expression]] = None) -> Expression:
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
        >>> print(e)
        x**2 + 5

        Define a regular symbol and use it as a function:
        >>> f = S('f')
        >>> e = f(1,2)
        >>> print(e)
        f(1,2)


        Define a symmetric function:
        >>> f = S('f', is_symmetric=True)
        >>> e = f(2,1)
        >>> print(e)
        f(1,2)


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

        Parameters
        ----------
        name : str
            The name of the symbol
        is_symmetric : Optional[bool]
            Set to true if the symbol is symmetric.
        is_antisymmetric : Optional[bool]
            Set to true if the symbol is antisymmetric.
        is_cyclesymmetric : Optional[bool]
            Set to true if the symbol is cyclesymmetric.
        is_linear : Optional[bool]
            Set to true if the symbol is linear.
        is_scalar : Optional[bool]
            Set to true if the symbol is a scalar. It will be moved out of linear functions.
        is_real : Optional[bool]
            Set to true if the symbol is a real number.
        is_integer : Optional[bool]
            Set to true if the symbol is an integer.
        is_positive : Optional[bool]
            Set to true if the symbol is a positive number.
        tags: Optional[Sequence[str]]
            A list of tags to associate with the symbol.
        custom_normalization : Optional[Transformer]
            A transformer that is called after every normalization. Note that the symbol
            name cannot be used in the transformer as this will lead to a definition of the
            symbol. Use a wildcard with the same attributes instead.
        custom_print : Optional[Callable[..., Optional[str]]]:
            A function that is called when printing the variable/function, which is provided as its first argument.
            This function should return a string, or `None` if the default print function should be used.
            The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
        custom_derivative: Optional[Callable[[Expression, int], Expression]]:
            A function that is called when computing the derivative of a function in a given argument.
        """

    @overload
    @classmethod
    def symbol(_cls,
               *names: str,
               is_symmetric: Optional[bool] = None,
               is_antisymmetric: Optional[bool] = None,
               is_cyclesymmetric: Optional[bool] = None,
               is_linear: Optional[bool] = None,
               is_real: Optional[bool] = None,
               is_scalar: Optional[bool] = None,
               is_integer: Optional[bool] = None,
               is_positive: Optional[bool] = None,
               tags: Optional[Sequence[str]] = None) -> Sequence[Expression]:
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
        >>> print(e)
        f(1,2)

        Parameters
        ----------
        name : str
            The name of the symbol
        is_symmetric : Optional[bool]
            Set to true if the symbol is symmetric.
        is_antisymmetric : Optional[bool]
            Set to true if the symbol is antisymmetric.
        is_cyclesymmetric : Optional[bool]
            Set to true if the symbol is cyclesymmetric.
        is_linear : Optional[bool]
            Set to true if the symbol is multilinear.
        is_scalar : Optional[bool]
            Set to true if the symbol is a scalar. It will be moved out of linear functions.
        is_real : Optional[bool]
            Set to true if the symbol is a real number.
        is_integer : Optional[bool]
            Set to true if the symbol is an integer.
        is_positive : Optional[bool]
            Set to true if the symbol is a positive number.
        tags: Optional[Sequence[str]]
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
        >>> print(e)
        f(3,x)
        """

    @overload
    def __call__(self, *args: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Create a Symbolica expression or transformer by calling the function with appropriate arguments.

        Examples
        -------
        >>> x, f = S('x', 'f')
        >>> e = f(3,x)
        >>> print(e)
        f(3,x)
        """

    @classmethod
    def num(_cls, num: int | float | complex | str | Decimal, relative_error: Optional[float] = None) -> Expression:
        """Create a new Symbolica number from an int, a float, or a string.
        A floating point number is kept as a float with the same precision as the input,
        but it can also be converted to the smallest rational number given a `relative_error`.

        Examples
        --------
        >>> e = Expression.num(1) / 2
        >>> print(e)
        1/2

        >>> print(Expression.num(1/3))
        >>> print(Expression.num(0.33, 0.1))
        >>> print(Expression.num('0.333`3'))
        >>> print(Expression.num(Decimal('0.1234')))
        3.3333333333333331e-1
        1/3
        3.33e-1
        1.2340e-1
        """

    @classmethod
    def get_all_symbol_names(_cls) -> list[str]:
        """Return all defined symbol names (function names and variables)."""

    @classmethod
    def parse(_cls, input: str, mode: ParseMode = ParseMode.Symbolica, default_namespace: str = "python") -> Expression:
        """
        Parse a Symbolica expression from a string.

        Parameters
        ----------
        input: str
            An input string. UTF-8 characters are allowed.
        mode: ParseMode
            The parsing mode to use. Use `ParseMode.Mathematica` to parse Mathematica expressions.
        default_namespace: str
            The default namespace to use when parsing symbols.

        Examples
        --------
        >>> e = E('x^2+y+y*4')
        >>> print(e)

        `x^2+5*y`

        >>> e = E('Cos[test`x] (2+ 3 I)', mode=ParseMode.Mathematica)
        >>> print(e)

        `cos(test::x)(2+3i)`

        Raises
        ------
        ValueError
            If the input is not a valid expression.
        """

    def __new__(cls) -> Expression:
        """Create a new expression that represents 0."""

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
    def load(_cls, filename: str, conflict_fn: Optional[Callable[[str], str]] = None) -> Expression:
        """Load an expression and its state from a file. The state will be merged
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
        """

    def save(self, filename: str, compression_level: int = 9):
        """Save the expression and its state to a binary file.
        The data is compressed and the compression level can be set between 0 and 11.

        The data can be loaded using `Expression.load`.

        Examples
        --------
        >>> e = E("f(x)+f(y)").expand()
        >>> e.save('export.dat')
        """

    def get_byte_size(self) -> int:
        """ Get the number of bytes that this expression takes up in memory."""

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = 100,
        custom_print_mode: Optional[int] = None,
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

    def to_sympy(self) -> str:
        """Convert the expression into a sympy-parsable string.

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
        """

    def __hash__(self) -> str:
        """
        Hash the expression.
        """

    def get_type(self) -> AtomType:
        """Get the type of the atom."""

    def to_atom_tree(self) -> AtomTree:
        """Convert the expression to a tree."""

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

    def is_scalar(self) -> bool:
        """
        Check if the expression is a scalar. Symbols must have the scalar attribute.

        Examples
        --------
        >>> x = S('x', is_scalar=True)
        >>> e = (x + 1)**2 + 5
        >>> print(e.is_scalar())
        True
        """

    def is_real(self) -> bool:
        """
        Check if the expression is real. Symbols must have the real attribute.

        Examples
        --------
        >>> x = S('x', is_real=True)
        >>> e = (x + 1)**2 / 2 + 5
        >>> print(e.is_real())
        True
        """

    def is_integer(self) -> bool:
        """
        Check if the expression is integer. Symbols must have the integer attribute.

        Examples
        --------
        >>> x = S('x', is_integer=True)
        >>> e = (x + 1)**2 + 5
        >>> print(e.is_integer())
        True
        """

    def is_positive(self) -> bool:
        """
        Check if the expression is a positive scalar. Symbols must have the positive attribute.

        Examples
        --------
        >>> x = S('x', is_positive=True)
        >>> e = (x + 1)**2 + 5
        >>> print(e.is_positive())
        True
        """

    def is_finite(self) -> bool:
        """
        Check if the expression has no infinities and is not indeterminate.

        Examples
        --------
        >>> e = E('1/x + x^2 + log(0)')
        >>> print(e.is_finite())
        False
        """

    def is_constant(self) -> bool:
        """
        Check if the expression is constant, i.e. contains no user-defined symbols or functions.

        Examples
        --------
        >>> e = E('cos(2 + exp(3)) + 5')
        >>> print(e.is_constant())
        True
        """

    def __add__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __radd__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Add this expression to `other`, returning the result.
        """

    def __sub__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Subtract `other` from this expression, returning the result.
        """

    def __rsub__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Subtract this expression from `other`, returning the result.
        """

    def __mul__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Multiply this expression with `other`, returning the result.
        """

    def __rmul__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Multiply this expression with `other`, returning the result.
        """

    def __truediv__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Divide this expression by `other`, returning the result.
        """

    def __rtruediv__(self, other: Expression | int | float | complex | Decimal) -> Expression:
        """
        Divide `other` by this expression, returning the result.
        """

    def __pow__(self, exp: Expression | int | float | complex | Decimal) -> Expression:
        """
        Take `self` to power `exp`, returning the result.
        """

    def __rpow__(self, base: Expression | int | float | complex | Decimal) -> Expression:
        """
        Take `base` to power `self`, returning the result.
        """

    def __xor__(self, a: Any) -> Expression:
        """
        Returns a warning that `**` should be used instead of ` ^ ` for taking a power.
        """

    def __rxor__(self, a: Any) -> Expression:
        """
        Returns a warning that `**` should be used instead of ` ^ ` for taking a power.
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
        """

    def contains(self, a: Transformer | HeldExpression | Expression | int | float | Decimal) -> Condition:
        """Returns true iff `self` contains `a` literally.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y, z = S('x', 'y', 'z')
        >>> e = x * y * z
        >>> e.contains(x) # True
        >>> e.contains(x*y*z) # True
        >>> e.contains(x*y) # False
        """

    def get_all_symbols(self, include_function_symbols: bool = True) -> Sequence[Expression]:
        """ Get all symbols in the current expression, optionally including function symbols.
        The symbols are sorted in Symbolica's internal ordering.
        """

    def get_all_indeterminates(self, enter_functions: bool = True) -> Sequence[Expression]:
        """ Get all symbols and functions in the current expression, optionally including function symbols.
        The symbols are sorted in Symbolica's internal ordering.
        """

    def to_float(self, decimal_prec: int = 16) -> Expression:
        """Convert all coefficients and built-in functions to floats with a given precision `decimal_prec`.
        The precision of floating point coefficients in the input will be truncated to `decimal_prec`."""

    def rationalize(self, relative_error: float = 0.01) -> Expression:
        """Map all floating point and rational coefficients to the best rational approximation
        in the interval `[self*(1-relative_error),self*(1+relative_error)]`."""

    def req_len(self, min_length: int, max_length: int | None) -> PatternRestriction:
        """
        Create a pattern restriction based on the wildcard length before downcasting.
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
        >>> print(e)

        Yields `1`.
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
        >>> print(e)

        Yields `f(x)`.
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
        >>> print(e)

        Yields `f(x)*f(1)`.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the expression is of a certain type.
        """

    def req_contains(self, a: Expression) -> PatternRestriction:
        """
        Create a pattern restriction that filters for expressions that contain `a`.
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
        """

    def req_lt(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
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
        """

    def req_gt(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than a number `num`.
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
        """

    def req_le(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
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
        """

    def req_ge(self, num: Expression | int | float | complex | Decimal, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
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
        """

    def req_cmp_lt(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than another wildcard.
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
        """

    def req_cmp_gt(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than another wildcard.
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
        """

    def req_cmp_le(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is smaller than or equal to another wildcard.
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
        """

    def req_cmp_ge(self, num: Expression, cmp_any_atom=False) -> PatternRestriction:
        """Create a pattern restriction that passes when the wildcard is greater than or equal to another wildcard.
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
        """

    def __eq__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions.
        """

    def __neq__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions.
        """

    def __lt__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __le__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __gt__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __ge__(self, other: Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two expressions. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __iter__(self) -> Iterator[Expression]:
        """
        Create an iterator over all atoms in the expression.
        """

    def __getitem__(self, idx: int) -> Expression:
        """Get the `idx`th component of the expression."""

    def map(
        self,
        transformations: Transformer,
        n_cores: Optional[int] = 1,
        stats_to_file: Optional[str] = None,
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
            The number of cores to use for parallel execution.
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

    def expand(self, var: Optional[Expression] = None, via_poly: Optional[bool] = None) -> Expression:
        """
        Expand the expression. Optionally, expand in `var` only.

        Using `via_poly=True` may give a significant speedup for large expressions.
        """

    def expand_num(self) -> Expression:
        """ Distribute numbers in the expression, for example: `2*(x+y)` -> `2*x+2*y`.

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
        key_map: Optional[Callable[[Expression], Expression]] = None,
        coeff_map: Optional[Callable[[Expression], Expression]] = None,
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
        >>> print(e.collect(x))

        yields `x^2+x*(y+5)+5`.

        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> var, coeff = Expression.funs('var', 'coeff')
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
        key_map: Optional[Callable[[Expression], Expression]] = None,
        coeff_map: Optional[Callable[[Expression], Expression]] = None,
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
        >>> print(e.collect_symbol(f))

        yields `(1+x)*f(1,2)`.

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
        """Collect common factors from (nested) sums.

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

    def collect_num(self) -> Expression:
        """Collect numerical factors by removing the content from additions.
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
    ) -> Sequence[Tuple[Expression, Expression]]:
        """Collect terms involving the same power of `x`, where `x` are variables or functions.
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
        """

    def coefficient(self, x: Expression) -> Expression:
        """Collect terms involving the literal occurrence of `x`.

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
        """

    def derivative(self, x: Expression) -> Expression:
        """Derive the expression w.r.t the variable `x`."""

    def series(
        self,
        x: Expression,
        expansion_point: Expression | int | float | complex | Decimal,
        depth: int,
        depth_denom: int = 1,
        depth_is_absolute: bool = True
    ) -> Series:
        """Series expand in `x` around `expansion_point` to depth `depth`."""

    def apart(self, x: Optional[Expression] = None) -> Expression:
        """Compute the partial fraction decomposition in `x`.

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
        """

    def together(self) -> Expression:
        """Write the expression over a common denominator.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('v1^2/2+v1^3/v4*v2+v3/(1+v4)')
        >>> print(p.together())
        """

    def cancel(self) -> Expression:
        """Cancel common factors between numerators and denominators.
        Any non-canceling parts of the expression will not be rewritten.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('1+(y+1)^10*(x+1)/(x^2+2x+1)')
        >>> print(p.cancel())
        1+(y+1)**10/(x+1)
        """

    def factor(self) -> Expression:
        """Factor the expression over the rationals.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(6 + x)/(7776 + 6480*x + 2160*x^2 + 360*x^3 + 30*x^4 + x^5)')
        >>> print(p.factor())
        (x+6)**-4
        """

    @overload
    def to_polynomial(self, vars: Optional[Sequence[Expression]] = None) -> Polynomial:
        """Convert the expression to a polynomial, optionally, with the variable ordering specified in `vars`.
        All non-polynomial parts will be converted to new, independent variables.
        """

    @overload
    def to_polynomial(self, minimal_poly: Polynomial, vars: Optional[Sequence[Expression]] = None,
                      ) -> NumberFieldPolynomial:
        """Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
        All non-polynomial elements will be converted to new independent variables.

        The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
        The minimal polynomial must be a monic, irreducible univariate polynomial.
        """

    @overload
    def to_polynomial(self,
                      modulus: int,
                      power: Optional[Tuple[int, Expression]] = None,
                      minimal_poly: Optional[Polynomial] = None,
                      vars: Optional[Sequence[Expression]] = None,
                      ) -> FiniteFieldPolynomial:
        """Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
        All non-polynomial elements will be converted to new independent variables.

        The coefficients will be converted to finite field elements modulo `modulus`.
        If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
        `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

        If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.
        """

    def to_rational_polynomial(
        self,
        vars: Optional[Sequence[Expression]] = None,
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
        """

    def match(
        self,
        lhs: Expression | int | float | complex | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
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
        """

    def matches(
        self,
        lhs: Expression | int | float | complex | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> Condition:
        """
        Test whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.

        Examples
        --------

        >>> f = S('f')
        >>> if f(1).matches(f(2)):
        >>>    print('match')
        """

    def replace_iter(
        self,
        lhs: Expression | int | float | complex | Decimal,
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
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
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth:
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        allow_new_wildcards_on_rhs:
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        """

    def replace(
        self,
        pattern: Expression | int | float | complex | Decimal,
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        non_greedy_wildcards: Optional[Sequence[Expression]] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
        rhs_cache_size: Optional[int] = None,
        repeat: Optional[bool] = False,
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
        cond:
            Conditions on the pattern.
        non_greedy_wildcards:
            Wildcards that try to match as little as possible.
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth: bool, optional
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        allow_new_wildcards_on_rhs: bool, optional
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        rhs_cache_size: int, optional
            Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
            **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
        repeat: bool, optional
            If set to `True`, the entire operation will be repeated until there are no more matches.
        """

    def replace_multiple(self, replacements: Sequence[Replacement],  repeat: Optional[bool] = False) -> Expression:
        """
        Replace all atoms matching the patterns. See `replace` for more information.

        The entire operation can be repeated until there are no more matches using `repeat=True`.

        Examples
        --------

        >>> x, y, f = S('x', 'y', 'f')
        >>> e = f(x,y)
        >>> r = e.replace_multiple([Replacement(x, y), Replacement(y, x)])
        >>> print(r)
        f(y,x)

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
        """Solve a linear system in the variables `variables`, where each expression
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
        """

    @overload
    def nsolve(
        self,
        variable: Expression,
        init: float,
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> float:
        """Find the root of an expression in `x` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = E("x^2-2").nsolve(E("x"), 1., 0.0001, 1000000)
        """

    @overload
    def nsolve(
        self,
        variable: Expression,
        init: Decimal,
        prec: float = 1e-4,
        max_iter: int = 10000,
    ) -> Decimal:
        """Find the root of an expression in `x` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = E("x^2-2").nsolve(E("x"),
                      Decimal("1.000000000000000000000000000000000000000000000000000000000000000000000000"), 1e-74, 1000000)
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
        """Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = Expression.nsolve_system([E("5x^2+x*y^2+sin(2y)^2 - 2"), E("exp(2x-y)+4y - 3")], [S("x"), S("y")],
                             [1., 1.], 1e-20, 1000000)
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
        """Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
        Use `init` as the initial guess for the root. This method uses the same precision as `init`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = Expression.nsolve_system([E("5x^2+x*y^2+sin(2y)^2 - 2"), E("exp(2x-y)+4y - 3")], [S("x"), S("y")],
                             [Decimal("1.00000000000000000"), Decimal("1.00000000000000000")], 1e-20, 1000000)
        """

    def evaluate(
        self, constants: dict[Expression, float], funs: dict[Expression, Callable[[Sequence[float]], float]]
    ) -> float:
        """Evaluate the expression, using a map of all the variables and
        user functions to a float.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> f = S('f')
        >>> e = E('cos(x)')*3 + f(x,2)
        >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))
        """

    def evaluate_with_prec(
        self,
        constants: dict[Expression, float | str | Decimal],
        funs: dict[Expression, Callable[[Sequence[Decimal]], float | str | Decimal]],
        decimal_digit_precision: int
    ) -> Decimal:
        """Evaluate the expression, using a map of all the constants and
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
        """

    def evaluate_complex(
        self, constants: dict[Expression, float | complex], funs: dict[Expression, Callable[[Sequence[complex]], float | complex]]
    ) -> complex:
        """Evaluate the expression, using a map of all the variables and
        user functions to a complex number.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> e = E('sqrt(x)')*y
        >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))
        """

    def evaluator(
        self,
        constants: dict[Expression, Expression],
        funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression],
        params: Sequence[Expression],
        iterations: int = 100,
        n_cores: int = 4,
        verbose: bool = False,
        external_functions: Optional[dict[Tuple[Expression, str], Callable[[
            Sequence[float | complex]], float | complex]]] = None,
        conditionals: Optional[Sequence[Expression]] = None,
    ) -> Evaluator:
        """Create an evaluator that can evaluate (nested) expressions in an optimized fashion.
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
        funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression]
            A dictionary of functions. The key is a tuple of the function name, printable name and the argument variables.
            The value is the function body.
        params: Sequence[Expression]
            A list of free parameters.
        iterations: int, optional
            The number of optimization iterations to perform.
        n_cores: int, optional
            The number of cores to use for the optimization.
        verbose: bool, optional
            Print the progress of the optimization.
        external_functions: Optional[dict[Tuple[Expression, str], Callable[[Sequence[float | complex]], float | complex]]]
            A dictionary of external functions that can be called during evaluation.
            The key is the function name and the value is a callable that takes a list of arguments and returns a float.
            This is useful for functions that are not defined in Symbolica but are available in Python.
        conditionals: Optional[Sequence[Expression]], optional
            A list of conditional functions. These functions should take three argument: a condition that is tested for
            inequality with 0, the true branch and the false branch.
        """

    @classmethod
    def evaluator_multiple(
        _cls,
        exprs: Sequence[Expression],
        constants: dict[Expression, Expression],
        funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression],
        params: Sequence[Expression],
        iterations: int = 100,
        n_cores: int = 4,
        verbose: bool = False,
    ) -> Evaluator:
        """Create an evaluator that can jointly evaluate (nested) expressions in an optimized fashion.
        See `Expression.evaluator()` for more information.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> e1 = E("x^2 + 1")
        >>> e2 = E("x^2 + 2)
        >>> ev = Expression.evaluator_multiple([e1, e2], {}, {}, [x])

        will recycle the `x^2`
        """

    def canonize_tensors(self,
                         contracted_indices: Sequence[Tuple[Expression | int, Expression | int]]) -> Tuple[Expression, list[Tuple[Expression, Expression]], list[Tuple[Expression, Expression]]]:
        """Canonize (products of) tensors in the expression by relabeling repeated indices.
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
        """


class Replacement:
    """A replacement of a pattern by a right-hand side."""

    def __new__(
            cls,
            pattern: Expression | int | float | complex | Decimal,
            rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
            cond: Optional[PatternRestriction | Condition] = None,
            non_greedy_wildcards: Optional[Sequence[Expression]] = None,
            level_range: Optional[Tuple[int, Optional[int]]] = None,
            level_is_tree_depth: Optional[bool] = False,
            allow_new_wildcards_on_rhs: Optional[bool] = False,
            rhs_cache_size: Optional[int] = None) -> Replacement:
        """Create a new replacement. See `replace` for more information."""


class PatternRestriction:
    """A restriction on wildcards."""

    def __and__(self, other: PatternRestriction) -> PatternRestriction:
        """Create a new pattern restriction that is the logical and operation between two restrictions (i.e., both should hold)."""

    def __or__(self, other: PatternRestriction) -> PatternRestriction:
        """Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold)."""

    def __invert__(self) -> PatternRestriction:
        """Create a new pattern restriction that takes the logical 'not' of the current restriction."""

    @classmethod
    def req_matches(_cls, match_fn: Callable[[dict[Expression, Expression]], int]) -> PatternRestriction:
        """Create a pattern restriction based on the current matched variables.
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
        """


class Condition:
    """Relations that evaluate to booleans"""

    def eval(self) -> bool:
        """Evaluate the condition."""

    def __repr__(self) -> str:
        """Return a string representation of the condition."""

    def __str__(self) -> str:
        """Return a string representation of the condition."""

    def __bool__(self) -> bool:
        """Return the boolean value of the condition."""

    def __and__(self, other:  Condition) -> Condition:
        """Create a condition that is the logical and operation between two conditions (i.e., both should hold)."""

    def __or__(self, other:  Condition) -> Condition:
        """Create a condition that is the logical 'or' operation between two conditions (i.e., at least one of the two should hold)."""

    def __invert__(self) -> Condition:
        """Create a condition that takes the logical 'not' of the current condition."""

    def to_req(self) -> PatternRestriction:
        """Convert the condition to a pattern restriction."""


class CompareOp:
    """One of the following comparison operators: `<`,`>`,`<=`,`>=`,`==`,`!=`."""


class HeldExpression:
    def __call__(self) -> Expression:
        """Execute a bound transformer. If the transformer is unbound,
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
        """

    def __neq__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers.
        """

    def __lt__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __le__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __gt__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __ge__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the transformed expression is of a certain type.
        """

    def contains(self, element: HeldExpression | Expression | int | float | complex | Decimal) -> Condition:
        """
        Create a transformer that checks if the expression contains the given `element`.
        """

    def matches(
        self,
        lhs: Expression | int | float | complex | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> Condition:
        """
        Create a transformer that tests whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.
        """

    def __add__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.
        """

    def __radd__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.
        """

    def __sub__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Subtract `other` from this transformer, returning the result.
        """

    def __rsub__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Subtract this transformer from `other`, returning the result.
        """

    def __mul__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.
        """

    def __rmul__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Add this transformer to `other`, returning the result.
        """

    def __truediv__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Divide this transformer by `other`, returning the result.
        """

    def __rtruediv__(self, other: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Divide `other` by this transformer, returning the result.
        """

    def __pow__(self, exp: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Take `self` to power `exp`, returning the result.
        """

    def __rpow__(self, base: HeldExpression | Expression | int | float | complex | Decimal) -> HeldExpression:
        """
        Take `base` to power `self`, returning the result.
        """

    def __xor__(self, a: Any) -> HeldExpression:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.
        """

    def __rxor__(self, a: Any) -> HeldExpression:
        """
        Returns a warning that `**` should be used instead of `^` for taking a power.
        """

    def __neg__(self) -> HeldExpression:
        """
        Negate the current transformer, returning the result.
        """


class Transformer:
    """Operations that transform an expression."""

    def __new__(_cls) -> Transformer:
        """Create a new transformer for a term provided by `Expression.map`."""

    def __call__(self, expr: Expression | int | float | complex | Decimal, stats_to_file: Optional[str] = None) -> Expression:
        """Execute an unbound transformer on the given expression. If the transformer
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

    def if_then(self, condition: Condition, if_block: Transformer, else_block: Optional[Transformer] = None) -> Transformer:
        """Evaluate the condition and apply the `if_block` if the condition is true, otherwise apply the `else_block`.
        The expression that is the input of the transformer is the input for the condition, the `if_block` and the `else_block`.

        Examples
        --------
        >>> t = T().map_terms(T().if_then(T().contains(x), T().print()))
        >>> t(x + y + 4)

        prints `x`.
        """

    def if_changed(self, condition: Transformer, if_block: Transformer, else_block: Optional[Transformer] = None) -> Transformer:
        """Execute the `condition` transformer. If the result of the `condition` transformer is different from the input expression,
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
        """

    def break_chain(self) -> Transformer:
        """Break the current chain and all higher-level chains containing `if` transformers.

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

    def expand(self, var: Optional[Expression] = None, via_poly: Optional[bool] = None) -> Transformer:
        """Create a transformer that expands products and powers. Optionally, expand in `var` only.

        Using `via_poly=True` may give a significant speedup for large expressions.

        Examples
        --------
        >>> from symbolica import *
        >>> x, x_ = S('x', 'x_')
        >>> f = S('f')
        >>> e = f((x+1)**2).replace(f(x_), x_.hold(T().expand()))
        >>> print(e)
        """

    def expand_num(self) -> Expression:
        """Create a transformer that distributes numbers in the expression, for example: `2*(x+y)` -> `2*x+2*y`.

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
        """Create a transformer that computes the product of a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(2,3).replace(f(x__), x__.hold(T().prod()))
        >>> print(e)
        """

    def sum(self) -> Transformer:
        """Create a transformer that computes the sum of a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(2,3).replace(f(x__), x__.hold(T().sum()))
        >>> print(e)
        """

    def nargs(self, only_for_arg_fun: bool = False) -> Transformer:
        """Create a transformer that returns the number of arguments.
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
        """

    def sort(self) -> Transformer:
        """Create a transformer that sorts a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(3,2,1).replace(f(x__), x__.hold(T().sort()))
        >>> print(e)
        """

    def cycle_symmetrize(self) -> Transformer:
        """ Create a transformer that cycle-symmetrizes a function.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x__')
        >>> f = S('f')
        >>> e = f(1,2,4,1,2,3).replace(f(x__), x_.hold(T().cycle_symmetrize()))
        >>> print(e)

        Yields `f(1,2,3,1,2,4)`.
        """

    def deduplicate(self) -> Transformer:
        """Create a transformer that removes elements from a list if they occur
        earlier in the list as well.

        Examples
        --------
        >>> from symbolica import *
        >>> x__ = S('x__')
        >>> f = S('f')
        >>> e = f(1,2,1,2).replace(f(x__), x__.hold(T().deduplicate()))
        >>> print(e)

        Yields `f(1,2)`.
        """

    def from_coeff(self) -> Transformer:
        """Create a transformer that extracts a rational polynomial from a coefficient.

        Examples
        --------
        >>> from symbolica import *, Function
        >>> e = Function.COEFF((x^2+1)/y^2).hold(T().from_coeff())
        >>> print(e)
        """

    def split(self) -> Transformer:
        """Create a transformer that split a sum or product into a list of arguments.

        Examples
        --------
        >>> from symbolica import *
        >>> x, x__ = S('x', 'x__')
        >>> f = S('f')
        >>> e = (x + 1).replace(x__, f(x_.hold(T().split())))
        >>> print(e)
        """

    def linearize(self, symbols: Optional[Sequence[Expression]]) -> Transformer:
        """Create a transformer that linearizes a function, optionally extracting `symbols`
        as well.

        Examples
        --------

        >>> from symbolica import *
        >>> x, y, z, w, f, x__ = S('x', 'y', 'z', 'w', 'f', 'x__')
        >>> e = f(x+y, 4*z*w+3).replace(f(x__), f(x__).hold(T().linearize([z])))
        >>> print(e)

        yields `f(x,3)+f(y,3)+4*z*f(x,w)+4*z*f(y,w)`.
        """

    def partitions(
        self,
        bins: Sequence[Tuple[Transformer | Expression, int]],
        fill_last: bool = False,
        repeat: bool = False,
    ) -> Transformer:
        """Create a transformer that partitions a list of arguments into named bins of a given length,
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
        """

    def permutations(self, function_name: Transformer | Expression) -> Transformer:
        """Create a transformer that generates all permutations of a list of arguments.

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
        """

    def map(self, f: Callable[[Expression], Expression | int | float | complex | Decimal]) -> Transformer:
        """Create a transformer that applies a Python function.

        Examples
        --------
        >>> from symbolica import *
        >>> x_ = S('x_')
        >>> f = S('f')
        >>> e = f(2).replace(f(x_), x_.hold(T().map(lambda r: r**2)))
        >>> print(e)
        """

    def map_terms(self, *transformers: Transformer, n_cores: int = 1) -> Transformer:
        """Map a chain of transformer over the terms of the expression, optionally using multiple cores.

        Examples
        --------
        >>> from symbolica import *
        >>> x, y = S('x', 'y')
        >>> t = T().map_terms(T().print(), n_cores=2)
        >>> e = t(x + y)
        """

    def for_each(self, *transformers: Transformer) -> Transformer:
        """Create a transformer that applies a transformer chain to every argument of the `arg()` function.
        If the input is not `arg()`, the transformer is applied to the input.

        Examples
        --------
        >>> from symbolica import *
        >>> x = S('x')
        >>> f = S('f')
        >>> t = T().split().for_each(T().map(f))
        >>> e = t(1+x)
        """

    def check_interrupt(self) -> Transformer:
        """Create a transformer that checks for a Python interrupt,
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
        """Create a transformer that repeatedly executes the arguments in order
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
        """

    def chain(self, *transformers: Transformer) -> Transformer:
        """Chain several transformers. `chain(A,B,C)` is the same as `A.B.C`,
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
        """

    def derivative(self, x: HeldExpression | Expression) -> Transformer:
        """Create a transformer that derives `self` w.r.t the variable `x`."""

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
        key_map: Optional[Transformer] = None,
        coeff_map: Optional[Transformer] = None,
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
        >>> print(e.hold(T().collect(x).execute()))

        yields `x^2+x*(y+5)+5`.

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
        key_map: Optional[Callable[[Expression], Expression]] = None,
        coeff_map: Optional[Callable[[Expression], Expression]] = None,
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
        >>> print(T().collect_symbol(x)(e))

        yields `(1+x)*f(1,2)`.

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
        """Create a transformer that collects common factors from (nested) sums.

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

    def collect_num(self) -> Transformer:
        """Create a transformer that collects numerical factors by removing the content from additions.
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
        """Complex conjugate the expression."""

    def coefficient(self, x: Expression) -> Transformer:
        """Create a transformer that collects terms involving the literal occurrence of `x`.
        """

    def apart(self, x: Expression) -> Transformer:
        """Create a transformer that computes the partial fraction decomposition in `x`.
        """

    def together(self) -> Transformer:
        """Create a transformer that writes the expression over a common denominator.
        """

    def cancel(self) -> Transformer:
        """Create a transformer that cancels common factors between numerators and denominators.
        Any non-canceling parts of the expression will not be rewritten.
        """

    def factor(self) -> Transformer:
        """Create a transformer that factors the expression over the rationals."""

    def series(
        self,
        x: Expression,
        expansion_point: Expression | int | float | complex | Decimal,
        depth: int,
        depth_denom: int = 1,
        depth_is_absolute: bool = True
    ) -> Transformer:
        """Create a transformer that series expands in `x` around `expansion_point` to depth `depth`.

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
        """

    def replace(
        self,
        pat: HeldExpression | Expression | int | float | complex | Decimal,
        rhs: HeldExpression | Expression | Callable[[dict[Expression, Expression]], Expression] | int | float | complex | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        non_greedy_wildcards: Optional[Sequence[Expression]] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
        rhs_cache_size: Optional[int] = None,
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
        level_range:
            Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
        level_is_tree_depth:
            If set to `True`, the level is increased when going one level deeper in the expression tree.
        allow_new_wildcards_on_rhs:
            If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
        rhs_cache_size: int, optional
            Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
            **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
        """

    def replace_multiple(self, replacements: Sequence[Replacement]) -> Transformer:
        """
        Create a transformer that replaces all atoms matching the patterns. See `replace` for more information.

        Examples
        --------

        >>> x, y, f = S('x', 'y', 'f')
        >>> t = T().replace_multiple([Replacement(x, y), Replacement(y, x)])
        >>> r = t(f(x,y))
        >>> print(r)
        """

    def print(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = None,
        custom_print_mode: Optional[int] = None,
    ) -> Transformer:
        """
        Create a transformer that prints the expression.

        Examples
        --------
        >>> T().print(terms_on_new_line = True)
        """

    def stats(
        self,
        tag: str,
        transformer: Transformer,
        color_medium_change_threshold: Optional[float] = 10.,
        color_large_change_threshold: Optional[float] = 100.,
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
        """

    def __eq__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers.
        """

    def __neq__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers.
        """

    def __lt__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __le__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __gt__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def __ge__(self, other: Transformer | Expression | int | float | Decimal) -> Condition:
        """
        Compare two transformers. If any of the two expressions is not a rational number, an interal ordering is used.
        """

    def is_type(self, atom_type: AtomType) -> Condition:
        """
        Test if the transformed expression is of a certain type.
        """

    def contains(self, element: Transformer | HeldExpression | Expression | int | float | Decimal) -> Condition:
        """
        Create a transformer that checks if the expression contains the given `element`.
        """

    def matches(
        self,
        lhs: HeldExpression | Expression | int | float | Decimal,
        cond: Optional[PatternRestriction | Condition] = None,
        level_range: Optional[Tuple[int, Optional[int]]] = None,
        level_is_tree_depth: Optional[bool] = False,
        allow_new_wildcards_on_rhs: Optional[bool] = False,
    ) -> Condition:
        """
        Create a transformer that tests whether the pattern is found in the expression.
        Restrictions on the pattern can be supplied through `cond`.
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

    def __str__(self) -> str:
        """Print the series in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the series into a LaTeX string."""

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        precision: Optional[int] = None,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = None,
        custom_print_mode: Optional[int] = None,
    ) -> str:
        """
        Convert the series into a human-readable string.
        """

    def __add__(self, other: Series | Expression) -> Series:
        """Add another series or expression to this series, returning the result."""

    def __radd__(self, other: Expression) -> Series:
        """Add two series together, returning the result."""

    def __sub__(self, other: Series | Expression) -> Series:
        """Subtract `other` from `self`, returning the result."""

    def __rsub__(self, other: Expression) -> Series:
        """Subtract `self` from `other`, returning the result."""

    def __mul__(self, other: Series | Expression) -> Series:
        """Multiply another series or expression to this series, returning the result."""

    def __rmul__(self, other: Expression) -> Series:
        """Multiply two series together, returning the result."""

    def __truediv__(self, other: Series | Expression) -> Series:
        """Divide `self` by `other`, returning the result."""

    def __rtruediv__(self, other: Expression) -> Series:
        """Divide `other` by `self`, returning the result."""

    def __pow__(self, exp: int) -> Series:
        """Raise the series to the power of `exp`, returning the result."""

    def __neg__(self) -> Series:
        """Negate the series."""

    def sin(self) -> Series:
        """Compute the sine of the series, returning the result."""

    def cos(self) -> Series:
        """Compute the cosine of the series, returning the result."""

    def exp(self) -> Series:
        """Compute the exponential of the series, returning the result."""

    def log(self) -> Series:
        """Compute the natural logarithm of the series, returning the result."""

    def pow(self, num: int, den: int) -> Series:
        """Raise the series to the power of `num/den`, returning the result."""

    def spow(self, exp: Series) -> Series:
        """Raise the series to the power of `exp`, returning the result."""

    def shift(self, e: int) -> Series:
        """Shift the series by `e` units of the ramification."""

    def get_ramification(self) -> int:
        """Get the ramification."""

    def get_trailing_exponent(self) -> Tuple[int, int]:
        """Get the trailing exponent; the exponent of the first non-zero term."""

    def get_relative_order(self) -> Tuple[int, int]:
        """Get the relative order."""

    def get_absolute_order(self) -> Tuple[int, int]:
        """Get the absolute order."""

    def to_expression(self) -> Expression:
        """Convert the series to an expression"""


class TermStreamer:
    """
    A term streamer that can handle large expressions, by
    streaming terms to and from disk.
    """

    def __new__(_cls, path: Optional[str] = None,
                max_mem_bytes: Optional[int] = None,
                n_cores: Optional[int] = None) -> TermStreamer:
        """Create a new term streamer with a given path for its files,
           the maximum size of the memory buffer and the number of cores.
        """

    def __add__(self, other: TermStreamer) -> TermStreamer:
        """Add two term streamers together, returning the result."""

    def __iadd__(self, other: TermStreamer) -> None:
        """Add another term streamer to this one."""

    def clear(self) -> None:
        """Clear the term streamer."""

    def load(self, filename: str, conflict_fn: Optional[Callable[[str], str]] = None) -> int:
        """Load terms and their state from a binary file into the term streamer. The number of terms loaded is returned.

        The state will be merged with the current one. If a symbol has conflicting attributes, the conflict
        can be resolved using the renaming function `conflict_fn`.

        A term stream can be exported using `TermStreamer.save`.
        """

    def save(self, filename: str, compression_level: int = 9) -> None:
        """Export terms and their state to a binary file.
        The resulting file can be read back using `TermStreamer.load` or
        by using `Expression.load`. In the latter case, the whole term stream will be read into memory
        as a single expression.
        """

    def get_byte_size(self) -> int:
        """Get the byte size of the term stream."""

    def get_num_terms(self) -> int:
        """Get the number of terms in the stream."""

    def fits_in_memory(self) -> bool:
        """Check if the term stream fits in memory."""

    def push(self, expr: Expression) -> None:
        """Push an expression to the term stream."""

    def normalize(self) -> None:
        """Sort and fuse all terms in the stream."""

    def to_expression(self) -> Expression:
        """Convert the term stream into an expression. This may exceed the available memory."""

    def map(self, f: Transformer, stats_to_file: Optional[str] = None) -> TermStreamer:
        """Apply a transformer to all terms in the stream.

        Parameters
        ----------
        f: Transformer
            The transformer to apply.
        stats_to_file: str, optional
            If set, the output of the `stats` transformer will be written to a file in JSON format.
        """

    def map_single_thread(self, f: Transformer, stats_to_file: Optional[str] = None) -> TermStreamer:
        """Apply a transformer to all terms in the stream using a single thread.

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
        """Create the iterator."""

    def __next__(self) -> dict[Expression, Expression]:
        """Return the next match."""


class ReplaceIterator:
    """An iterator over single replacements."""

    def __iter__(self) -> ReplaceIterator:
        """Create the iterator."""

    def __next__(self) -> Expression:
        """Return the next replacement."""


class Polynomial:
    """A Symbolica polynomial with rational coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], default_namespace: str = "python") -> Polynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'])

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> Polynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        precision: Optional[int] = None,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = None,
        custom_print_mode: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_variables(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are equal."""

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are not equal."""

    def __lt__(self, rhs: int) -> bool:
        """Check if the polynomial is less than an integer."""

    def __le__(self, rhs: int) -> bool:
        """Check if the polynomial is less than or equal to an integer."""

    def __gt__(self, rhs: int) -> bool:
        """Check if the polynomial is greater than an integer."""

    def __ge__(self, rhs: int) -> bool:
        """Check if the polynomial is greater than or equal to an integer."""

    def __add__(self, rhs: Polynomial | int) -> Polynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: Polynomial | int) -> Polynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: Polynomial | int) -> Polynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __radd__(self, rhs: Polynomial | int) -> Polynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __rsub__(self, rhs: Polynomial | int) -> Polynomial:
        """Subtract polynomials `self` from `rhs`, returning the result."""

    def __rmul__(self, rhs: Polynomial | int) -> Polynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs`, rounding down, returning the result."""

    def __truediv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: Polynomial) -> Tuple[Polynomial, Polynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: Polynomial) -> Polynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> Polynomial:
        """Negate the polynomial."""

    def __pow__(self, exp: int) -> Polynomial:
        """Raise the polynomial to the power of `exp`, returning the result."""

    def __contains__(self, var: Expression) -> bool:
        """Check if the polynomial contains the given variable."""

    def contains(self, var: Expression) -> bool:
        """Check if the polynomial contains the given variable."""

    def degree(self, var: Expression) -> int:
        """Get the degree of the polynomial in `var`."""

    def reorder(self, vars: Sequence[Expression]) -> None:
        """Reorder the polynomial in-place to use the given variable order."""

    def gcd(self, *rhs: Polynomial) -> Polynomial:
        """Compute the greatest common divisor (GCD) of two or more polynomials."""

    def extended_gcd(self, rhs: Polynomial) -> Tuple[Polynomial, Polynomial, Polynomial]:
        """Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
        such that `self * s + rhs * t = gcd(self, rhs)`.

        Examples
        --------

        >>> from symbolica import *
        >>> E('(1+x)(20+x)').to_polynomial().extended_gcd(E('x^2+2').to_polynomial())

        yields `(1, 1/67-7/402*x, 47/134+7/402*x)`.
        """

    def resultant(self, rhs: Polynomial, var: Expression) -> Polynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def to_finite_field(self, prime: int) -> FiniteFieldPolynomial:
        """Convert the coefficients of the polynomial to a finite field with prime `prime`."""

    def isolate_roots(self, refine: Optional[float | Decimal] = None) -> list[Tuple[Expression, Expression, int]]:
        """Isolate the real roots of the polynomial. The result is a list of intervals with rational bounds that contain exactly one root,
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
        """

    def approximate_roots(self, max_iterations: int, tolerance: float) -> list[Tuple[complex, int]]:
        """Approximate all complex roots of a univariate polynomial, given a maximal number of iterations
        and a given tolerance. Returns the roots and their multiplicity.

        Examples
        --------

        >>> p = E('x^10+9x^7+4x^3+2x+1').to_polynomial()
        >>> for (r, m) in p.approximate_roots(1000, 1e-10):
        >>>     print(r, m)
        """

    def factor_square_free(self) -> list[Tuple[Polynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[Polynomial, int]]:
        """Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> Polynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def integrate(self, x: Expression) -> Polynomial:
        """Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.integrate(x))
        """

    def content(self) -> Polynomial:
        """Get the content, i.e., the GCD of the coefficients.

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
        >>> print(p.primitive())

        Yields `2*x^2+x+3`.
        """

    def monic(self) -> Polynomial:
        """
        Get the monic part of the polynomial, i.e., the polynomial
        divided by its leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.monic())

        Yields `x^2+1/2*x+2/3`.
        """

    def lcoeff(self) -> Polynomial:
        """
        Get the leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
        >>> print(p)

        Yields `3`.
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]] = None) -> list[Tuple[list[int], Polynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    @classmethod
    def groebner_basis(_cls, system: Sequence[Polynomial], grevlex: bool = True, print_stats: bool = False) -> list[Polynomial]:
        """Compute the Groebner basis of a polynomial system.

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
        """

    def reduce(self, gs: Sequence[Polynomial], grevlex: bool = True) -> Polynomial:
        """Completely reduce the polynomial w.r.t the polynomials `gs`.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        Examples
        --------
        >>> E('y^2+x').to_polynomial().reduce([E('x').to_polynomial()])

        yields `y^2`
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('x*y+2*x+x^2')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def evaluate(self, input: npt.ArrayLike) -> float:
        """Evaluate the polynomial at point `input`.

        Examples
        --------

        >>> from symbolica import *
        >>> P('x*y+2*x+x^2').evaluate([2., 3.])

        Yields `14.0`.
        """

    def evaluate_complex(self, input: npt.ArrayLike) -> complex:
        """Evaluate the polynomial at point `input` with complex input.

        Examples
        --------

        >>> from symbolica import *
        >>> P('x*y+2*x+x^2').evaluate([2+1j, 3+2j])

        Yields `11+13j`.
        """

    def replace(self, x: Expression, v: Polynomial | int) -> Polynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> r = E('y+1').to_polynomial())
        >>> p.replace(x, r)
        """

    @classmethod
    def interpolate(_cls, x: Expression, sample_points: Sequence[Expression | int], values: Sequence[Polynomial]) -> Polynomial:
        """Perform Newton interpolation in the variable `x` given the sample points
        `sample_points` and the values `values`.

        Examples
        --------
        >>> x, y = S('x', 'y')
        >>> a = Polynomial.interpolate(
        >>>         x, [4, 5], [(y**2+5).to_polynomial(), (y**3).to_polynomial()])
        >>> print(a)

        yields `25-5*x+5*y^2-y^2*x-4*y^3+y^3*x`.
        """

    def to_number_field(self, min_poly: Polynomial) -> NumberFieldPolynomial:
        """Convert the coefficients of the polynomial to a number field defined by the minimal polynomial `min_poly`.

        Examples
        --------
        >>> from symbolica import *
        >>> a = P('a').to_number_field(P('a^2-2'))
        >>> print(a * a)

        Yields `2`.
        """

    def adjoin(self, min_poly: Polynomial, new_symbol: Optional[Expression] = None) -> tuple[Polynomial, Polynomial, Polynomial]:
        """Adjoin the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
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
        """

    def simplify_algebraic_number(self, min_poly: Polynomial) -> Polynomial:
        """Find the minimal polynomial for the algebraic number represented by this polynomial
        expressed in the number field defined by `minimal_poly`.

        Examples
        --------

        >>> from symbolica import *
        >>> (min_poly, rep2, rep23) = P('a^2-2').adjoin(P('b^2-3'))
        >>> rep2.simplify_algebraic_number(min_poly)

        Yields `b^2-2`.
        """


class NumberFieldPolynomial:
    """A Symbolica polynomial with rational coefficients."""

    def __copy__(self) -> NumberFieldPolynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        precision: Optional[int] = None,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = None,
        custom_print_mode: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldNumberFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_variables(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are equal."""

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are not equal."""

    def __lt__(self, rhs: int) -> bool:
        """Check if the polynomial is less than an integer."""

    def __le__(self, rhs: int) -> bool:
        """Check if the polynomial is less than or equal to an integer."""

    def __gt__(self, rhs: int) -> bool:
        """Check if the polynomial is greater than an integer."""

    def __ge__(self, rhs: int) -> bool:
        """Check if the polynomial is greater than or equal to an integer."""

    def __add__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __radd__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __rsub__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Subtract polynomials `self` from `rhs`, returning the result."""

    def __rmul__(self, rhs: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs`, rounding down, returning the result."""

    def __truediv__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: NumberFieldPolynomial) -> Tuple[NumberFieldPolynomial, NumberFieldPolynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> NumberFieldPolynomial:
        """Negate the polynomial."""

    def __pow__(self, exp: int) -> NumberFieldPolynomial:
        """Raise the polynomial to the power of `exp`, returning the result."""

    def __contains__(self, var: Expression) -> bool:
        """Check if the polynomial contains the given variable."""

    def contains(self, var: Expression) -> bool:
        """Check if the polynomial contains the given variable."""

    def degree(self, var: Expression) -> int:
        """Get the degree of the polynomial in `var`."""

    def reorder(self, vars: Sequence[Expression]) -> None:
        """Reorder the polynomial in-place to use the given variable order."""

    def gcd(self, *rhs: NumberFieldPolynomial) -> NumberFieldPolynomial:
        """Compute the greatest common divisor (GCD) of two or more polynomials."""

    def extended_gcd(self, rhs: NumberFieldPolynomial) -> Tuple[NumberFieldPolynomial, NumberFieldPolynomial, NumberFieldPolynomial]:
        """Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
        such that `self * s + rhs * t = gcd(self, rhs)`.
        """

    def resultant(self, rhs: NumberFieldPolynomial, var: Expression) -> NumberFieldPolynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def factor_square_free(self) -> list[Tuple[NumberFieldPolynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[NumberFieldPolynomial, int]]:
        """Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> NumberFieldPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def integrate(self, x: Expression) -> NumberFieldPolynomial:
        """Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.integrate(x))
        """

    def content(self) -> NumberFieldPolynomial:
        """Get the content, i.e., the GCD of the coefficients.

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
        >>> print(p.primitive())

        Yields `x^2+2*x+3`.
        """

    def monic(self) -> NumberFieldPolynomial:
        """
        Get the monic part of the polynomial, i.e., the polynomial
        divided by its leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.monic())

        Yields `x^2+1/2*x+2/3`.
        """

    def lcoeff(self) -> NumberFieldPolynomial:
        """
        Get the leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
        >>> print(p)

        Yields `3`.
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]] = None) -> list[Tuple[list[int], NumberFieldPolynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    @classmethod
    def groebner_basis(_cls, system: list[NumberFieldPolynomial], grevlex: bool = True, print_stats: bool = False) -> list[NumberFieldPolynomial]:
        """Compute the Groebner basis of a polynomial system.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        If `print_stats=True` intermediate statistics will be printed.
        """

    def reduce(self, gs: Sequence[Polynomial], grevlex: bool = True) -> NumberFieldPolynomial:
        """Completely reduce the polynomial w.r.t the polynomials `gs`.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        Examples
        --------
        >>> E('y^2+x').to_polynomial().reduce([E('x').to_polynomial()])

        yields `y^2`
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('x*y+2*x+x^2')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def replace(self, x: Expression, v: NumberFieldPolynomial | int) -> NumberFieldPolynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> r = E('y+1').to_polynomial())
        >>> p.replace(x, r)
        """

    def get_minimal_polynomial(self) -> Polynomial:
        """Get the minimal polynomial of the algebraic extension."""

    def to_polynomial(self) -> Polynomial:
        """Convert the number field polynomial to a rational polynomial."""


class FiniteFieldPolynomial:
    """A Symbolica polynomial with finite field coefficients."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], prime: int, default_namespace: str = "python") -> FiniteFieldPolynomial:
        """
        Parse a polynomial with integer coefficients from a string.
        The input must be written in an expanded format and a list of all
        the variables must be provided.

        If these requirements are too strict, use `Expression.to_polynomial()` or
        `RationalPolynomial.parse()` instead.

        Examples
        --------
        >>> e = FiniteFieldPolynomial.parse('18*x^2+y+y*4', ['x', 'y'], 17)

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica polynomial.
        """

    def __copy__(self) -> FiniteFieldPolynomial:
        """Copy the polynomial."""

    def __str__(self) -> str:
        """Print the polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the polynomial into a LaTeX string."""

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        terms_on_new_line: bool = False,
        color_top_level_sum: bool = True,
        color_builtin_symbols: bool = True,
        print_ring: bool = True,
        symmetric_representation_for_finite_field: bool = False,
        explicit_rational_polynomial: bool = False,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        precision: Optional[int] = None,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = None,
        custom_print_mode: Optional[int] = None,
    ) -> str:
        """
        Convert the polynomial into a human-readable string, with tunable settings.

        Examples
        --------
        >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
        >>> print(p.format(symmetric_representation_for_finite_field=True))

        Yields `z³⁴+x^(x+2)+y⁴+f(x,x²)+128_378_127_123 z^(2/3) w² x⁻¹ y⁻¹+3/5`.
        """

    def nterms(self) -> int:
        """Get the number of terms in the polynomial."""

    def get_variables(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are equal."""

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are not equal."""

    def __add__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Subtract polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __radd__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Add two polynomials `self` and `rhs`, returning the result."""

    def __rsub__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Subtract polynomials `self` from `rhs`, returning the result."""

    def __rmul__(self, rhs: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Multiply two polynomials `self` and `rhs`, returning the result."""

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs`, rounding down, returning the result."""

    def __truediv__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Divide the polynomial `self` by `rhs` if possible, returning the result."""

    def quot_rem(self, rhs: FiniteFieldPolynomial) -> Tuple[FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """Divide `self` by `rhs`, returning the quotient and remainder."""

    def __mod__(self, rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Compute the remainder of the division of `self` by `rhs`."""

    def __neg__(self) -> FiniteFieldPolynomial:
        """Negate the polynomial."""

    def __pow__(self, exp: int) -> FiniteFieldPolynomial:
        """Raise the polynomial to the power of `exp`, returning the result."""

    def __contains__(self, var: Expression) -> bool:
        """Check if the polynomial contains the given variable."""

    def contains(self, var: Expression) -> bool:
        """Check if the polynomial contains the given variable."""

    def degree(self, var: Expression) -> int:
        """Get the degree of the polynomial in `var`."""

    def reorder(self, vars: Sequence[Expression]) -> None:
        """Reorder the polynomial in-place to use the given variable order."""

    def gcd(self, *rhs: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Compute the greatest common divisor (GCD) of two or more polynomials."""

    def extended_gcd(self, rhs: FiniteFieldPolynomial) -> Tuple[FiniteFieldPolynomial, FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
        such that `self * s + rhs * t = gcd(self, rhs)`.

        Examples
        --------

        >>> from symbolica import *
        >>> E('(1+x)(20+x)').to_polynomial(modulus=5).extended_gcd(E('x^2+2').to_polynomial(modulus=5))

        yields `(1, 3+4*x, 3+x)`.
        """

    def to_integer_polynomial(self, symmetric_representation: bool = True) -> Polynomial:
        """Convert the polynomial to a polynomial with integer coefficients."""

    def resultant(self, rhs: FiniteFieldPolynomial, var: Expression) -> FiniteFieldPolynomial:
        """Compute the resultant of two polynomials with respect to the variable `var`."""

    def factor_square_free(self) -> list[Tuple[FiniteFieldPolynomial, int]]:
        """Compute the square-free factorization of the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial().to_finite_field(7)
        >>> print('Square-free factorization of {}:'.format(p))
        >>> for f, exp in p.factor_square_free():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def factor(self) -> list[Tuple[FiniteFieldPolynomial, int]]:
        """Factorize the polynomial.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial().to_finite_field(7)
        >>> print('Factorization of {}:'.format(p))
        >>> for f, exp in p.factor():
        >>>     print('\t({})^{}'.format(f, exp))
        """

    def derivative(self, x: Expression) -> FiniteFieldPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.derivative(x))
        """

    def integrate(self, x: Expression) -> FiniteFieldPolynomial:
        """Integrate the polynomial in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x^2+2').to_polynomial()
        >>> print(p.integrate(x))
        """

    def monic(self) -> FiniteFieldPolynomial:
        """
        Get the monic part of the polynomial, i.e., the polynomial
        divided by its leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('6x^2+3x+9').to_polynomial()
        >>> print(p.monic())

        Yields `x^2+1/2*x+3/2`.
        """

    def lcoeff(self) -> FiniteFieldPolynomial:
        """
        Get the leading coefficient.

        Examples
        --------
        >>> from symbolica import Expression as E
        >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
        >>> print(p)

        Yields `3`.
        """

    def coefficient_list(self, xs: Optional[Expression | Sequence[Expression]] = None) -> list[Tuple[list[int], FiniteFieldPolynomial]]:
        """Get the coefficient list, optionally in the variables `xs`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> for n, pp in p.coefficient_list(x):
        >>>     print(n, pp)
        """

    @classmethod
    def groebner_basis(_cls, system: list[FiniteFieldPolynomial], grevlex: bool = True, print_stats: bool = False) -> list[FiniteFieldPolynomial]:
        """Compute the Groebner basis of a polynomial system.

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
        """Completely reduce the polynomial w.r.t the polynomials `gs`.

        If `grevlex=True`, reverse graded lexicographical ordering is used,
        otherwise the ordering is lexicographical.

        Examples
        --------
        >>> E('y^2+x').to_polynomial().reduce([E('x').to_polynomial()])

        yields `y^2`
        """

    def evaluate(self, input: Sequence[int]) -> int:
        """Evaluate the polynomial at point `input`.

        Examples
        --------

        >>> from symbolica import *
        >>> P('x*y+2*x+x^2', modulus=5).evaluate([2, 3])

        Yields `4`.
        """

    def replace(self, x: Expression, v: FiniteFieldPolynomial | int) -> FiniteFieldPolynomial:
        """Replace the variable `x` with a polynomial `v`.

        Examples
        --------

        >>> from symbolica import *
        >>> p = E('x*y+2*x+x^2').to_polynomial()
        >>> r = E('y+1').to_polynomial())
        >>> p.replace(S('x'), r)
        """

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression."""

    def to_polynomial(self) -> Polynomial:
        """Convert a Galois field polynomial to a simple finite field polynomial."""

    def to_galois_field(self, min_poly: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Convert the coefficients of the polynomial to a Galois field defined by the minimal polynomial `min_poly`."""

    def get_minimal_polynomial(self) -> FiniteFieldPolynomial:
        """Get the minimal polynomial of the algebraic extension."""

    def get_modulus(self) -> int:
        """Get the modulus of the finite field."""

    def adjoin(self, b: FiniteFieldPolynomial, new_symbol: Optional[Expression] = None) -> Tuple[FiniteFieldPolynomial, FiniteFieldPolynomial, FiniteFieldPolynomial]:
        """Adjoin the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
        is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.

        `b`  must be irreducible over `R` and `R[a]`; this is not checked.

        If `new_symbol` is provided, the variable of the new extension will be renamed to it.
        Otherwise, the variable of the new extension will be the same as that of `b`.
        """

    def simplify_algebraic_number(self, min_poly: FiniteFieldPolynomial) -> FiniteFieldPolynomial:
        """Find the minimal polynomial for the algebraic number represented by this polynomial
        expressed in the number field defined by `minimal_poly`.
        """


class RationalPolynomial:
    """A Symbolica rational polynomial."""

    def __new__(_cls, num: Polynomial, den: Polynomial) -> RationalPolynomial:
        """Create a new rational polynomial from a numerator and denominator polynomial."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], default_namespace: str = "python") -> RationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = RationalPolynomial.parse('(3/4*x^2+y+y*4)/(1+x)', ['x', 'y'])

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __copy__(self) -> RationalPolynomial:
        """Copy the rational polynomial."""

    def __str__(self) -> str:
        """Print the rational polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the rational polynomial into a LaTeX string."""

    def get_variables(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def numerator(self) -> Polynomial:
        """Get the numerator."""

    def denominator(self) -> Polynomial:
        """Get the denominator."""

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """Check if two rational polynomials are equal."""

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """Check if two rational polynomials are not equal."""

    def __lt__(self, rhs: int) -> bool:
        """Check if the rational polynomial is less than an integer."""

    def __le__(self, rhs: int) -> bool:
        """Check if the rational polynomial is less than or equal to an integer."""

    def __gt__(self, rhs: int) -> bool:
        """Check if the rational polynomial is greater than an integer."""

    def __ge__(self, rhs: int) -> bool:
        """Check if the polynomial is greater than or equal to an integer."""

    def __add__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Add two rational polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Multiply two rational polynomials `self` and `rhs`, returning the result."""

    def __floordiv__(self, rhs: Polynomial) -> Polynomial:
        """Divide the polynomial `self` by `rhs`, rounding down, returning the result."""

    def __truediv__(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> RationalPolynomial:
        """Negate the rational polynomial."""

    def gcd(self, rhs: RationalPolynomial) -> RationalPolynomial:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""

    def to_finite_field(self, prime: int) -> FiniteFieldRationalPolynomial:
        """Convert the coefficients of the rational polynomial to a finite field with prime `prime`."""

    def to_expression(self) -> Expression:
        """ Convert the polynomial to an expression.

        Examples
        --------

        >>> from symbolica import *
        >>> e = E('(x*y+2*x+x^2)/(x^7+y+1)')
        >>> p = e.to_polynomial()
        >>> print((e - p.to_expression()).expand())
        """

    def derivative(self, x: Expression) -> RationalPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> print(p.derivative(x))
        """

    def apart(self, x: Optional[Expression] = None) -> List[RationalPolynomial]:
        """Compute the partial fraction decomposition in `x`.

        If `None` is passed, the expression will be decomposed in all variables
        which involves a potentially expensive Groebner basis computation.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)
        """


class FiniteFieldRationalPolynomial:
    """A Symbolica rational polynomial."""

    def __new__(_cls, num: FiniteFieldPolynomial, den: FiniteFieldPolynomial) -> FiniteFieldRationalPolynomial:
        """Create a new rational polynomial from a numerator and denominator polynomial."""

    @classmethod
    def parse(_cls, input: str, vars: Sequence[str], prime: int, default_namespace: str = "python") -> FiniteFieldRationalPolynomial:
        """
        Parse a rational polynomial from a string.
        The list of all the variables must be provided.

        If this requirements is too strict, use `Expression.to_polynomial()` instead.

        Examples
        --------
        >>> e = FiniteFieldRationalPolynomial.parse('3*x^2+y+y*4', ['x', 'y'], 17)

        Raises
        ------
        ValueError
            If the input is not a valid Symbolica rational polynomial.
        """

    def __eq__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are equal."""

    def __ne__(self, rhs: Polynomial | int) -> bool:
        """Check if two polynomials are not equal."""

    def __copy__(self) -> FiniteFieldRationalPolynomial:
        """Copy the rational polynomial."""

    def __str__(self) -> str:
        """Print the rational polynomial in a human-readable format."""

    def to_latex(self) -> str:
        """Convert the rational polynomial into a LaTeX string."""

    def get_variables(self) -> Sequence[Expression]:
        """Get the list of variables in the internal ordering of the polynomial."""

    def __add__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Add two rational polynomials `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Subtract rational polynomials `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Multiply two rational polynomials `self` and `rhs`, returning the result."""

    def __truediv__(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Divide the rational polynomial `self` by `rhs` if possible, returning the result."""

    def __neg__(self) -> FiniteFieldRationalPolynomial:
        """Negate the rational polynomial."""

    def gcd(self, rhs: FiniteFieldRationalPolynomial) -> FiniteFieldRationalPolynomial:
        """Compute the greatest common divisor (GCD) of two rational polynomials."""

    def get_modulus(self) -> int:
        """Get the modulus of the finite field."""

    def derivative(self, x: Expression) -> RationalPolynomial:
        """Take a derivative in `x`.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> print(p.derivative(x))
        """

    def apart(self, x: Optional[Expression] = None) -> List[FiniteFieldRationalPolynomial]:
        """Compute the partial fraction decomposition in `x`.

        If `None` is passed, the expression will be decomposed in all variables
        which involves a potentially expensive Groebner basis computation.

        Examples
        --------

        >>> from symbolica import *
        >>> x = S('x')
        >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
        >>> for pp in p.apart(x):
        >>>     print(pp)
        """


class Matrix:
    """A matrix with rational polynomial coefficients."""

    def __new__(cls, nrows: int, ncols: int) -> Matrix:
        """Create a new zeroed matrix with `nrows` rows and `ncols` columns."""

    @classmethod
    def identity(cls, nrows: int) -> Matrix:
        """Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere."""

    @classmethod
    def eye(cls, diag: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere."""

    @classmethod
    def vec(cls, entries: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """Create a new column vector from a list of scalars."""

    @classmethod
    def from_linear(cls, nrows: int, ncols: int, entries: Sequence[RationalPolynomial | Polynomial | Expression | int]) -> Matrix:
        """Create a new matrix from a 1-dimensional vector of scalars."""

    @classmethod
    def from_nested(cls, entries: Sequence[Sequence[RationalPolynomial | Polynomial | Expression | int]]) -> Matrix:
        """Create a new matrix from a 2-dimensional vector of scalars."""

    def nrows(self) -> int:
        """Get the number of rows in the matrix."""

    def ncols(self) -> int:
        """Get the number of columns in the matrix."""

    def is_zero(self) -> bool:
        """Return true iff every entry in the matrix is zero."""

    def is_diagonal(self) -> bool:
        """Return true iff every non- main diagonal entry in the matrix is zero."""

    def transpose(self) -> Matrix:
        """Return the transpose of the matrix."""

    def swap_rows(self, i: int, j: int, start: int = 0) -> None:
        """Swap rows `i` and `j` of the matrix in-place, starting from column `start`."""

    def swap_cols(self, i: int, j: int) -> None:
        """Swap columns `i` and `j` of the matrix in-place."""

    def inv(self) -> Matrix:
        """Return the inverse of the matrix, if it exists."""

    def det(self) -> RationalPolynomial:
        """Return the determinant of the matrix."""

    def solve(self, b: Matrix) -> Matrix:
        """Solve `A * x = b` for `x`, where `A` is the current matrix."""

    def solve_any(self, b: Matrix) -> Matrix:
        """Solve `A * x = b` for `x`, where `A` is the current matrix and return any solution if the
        system is underdetermined.
        """

    def row_reduce(self, max_col: int) -> int:
        """Row-reduce the first `max_col` columns of the matrix in-place using Gaussian elimination and return the rank."""

    def augment(self, b: Matrix) -> Matrix:
        """Augment the matrix with another matrix, e.g. create `[A B]` from matrix `A` and `B`.

        Returns an error when the matrices do not have the same number of rows.
        """

    def split_col(self, col: int) -> Tuple[Matrix, Matrix]:
        """Split the matrix into two matrices at column `col`."""

    def content(self) -> RationalPolynomial:
        """Get the content, i.e., the GCD of the coefficients."""

    def primitive_part(self) -> Matrix:
        """Construct the same matrix, but with the content removed."""

    def map(self, f: Callable[[RationalPolynomial], RationalPolynomial]) -> Matrix:
        """Apply a function `f` to every entry of the matrix."""

    def format(
        self,
        mode: PrintMode = PrintMode.Symbolica,
        pretty_matrix=True,
        number_thousands_separator: Optional[str] = None,
        multiplication_operator: str = "*",
        double_star_for_exponentiation: bool = False,
        square_brackets_for_function: bool = False,
        num_exp_as_superscript: bool = True,
        precision: Optional[int] = None,
        show_namespaces: bool = False,
        include_attributes: bool = False,
        max_terms: Optional[int] = None,
        custom_print_mode: Optional[int] = None,
    ) -> str:
        """
        Convert the matrix into a human-readable string, with tunable settings.
        """

    def to_latex(self) -> str:
        """Convert the matrix into a LaTeX string."""

    def __copy__(self) -> Matrix:
        """Copy the matrix."""

    def __getitem__(self, key: Tuple[int, int]) -> RationalPolynomial:
        """Get the entry at position `key` in the matrix."""

    def __str__(self) -> str:
        """Print the matrix in a human-readable format."""

    def __eq__(self, other: Matrix) -> bool:
        """Compare two matrices."""

    def __neq__(self, other: Matrix) -> bool:
        """Compare two matrices."""

    def __add__(self, rhs: Matrix) -> Matrix:
        """Add two matrices `self` and `rhs`, returning the result."""

    def __sub__(self, rhs: Matrix) -> Matrix:
        """Subtract matrix `rhs` from `self`, returning the result."""

    def __mul__(self, rhs: Matrix | RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply `self` and `rhs`, returning the result."""

    def __rmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply  `rhs` and `self`, returning the result."""

    def __matmul__(self, rhs: Matrix | RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply `self` and `rhs`, returning the result."""

    def __rmatmul__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Matrix multiply  `rhs` and `self`, returning the result."""

    def __truediv__(self, rhs: RationalPolynomial | Polynomial | Expression | int) -> Matrix:
        """Divide this matrix by scalar `rhs` and return the result."""

    def __neg__(self) -> Matrix:
        """Negate the matrix, returning the result."""


class Evaluator:
    """An optimized evaluator of an expression."""

    def get_instructions(self) -> Tuple[List[Tuple[str, Tuple[str, int], List[Tuple[str, int]]]], int, List[Expression]]:
        """Return the instructions for efficiently evaluating the expression, the length of the list
        of temporary variables, and the list of constants. This can be used to generate
        code for the expression evaluation in any programming language.

        There are four lists that are used in the evaluation instructions:
        - `param`: the list of input parameters.
        - `temp`: the list of temporary slots. The size of it is provided as the second return value.
        - `const`: the list of constants.
        - `out`: the list of outputs.

        The instructions are of the form:
        - `('add', ('out', 0), [('const', 1), ('param', 0)])` which means `out[0] = const[1] + param[0]`.
        - `('mul', ('out', 0), [('temp', 0), ('param', 0)])` which means `out[0] = temp[0] * param[0]`.
        - `('pow', ('out', 0), ('param', 0), -1)` which means `out[0] = param[0]^-1`.
        - `('powf', ('out', 0), ('param', 0), ('param', 1))` which means `out[0] = param[0]^param[1]`.
        - `('fun', ('temp', 1), cos, ('param', 0))` which means `temp[1] = cos(param[0])`.
        - `('external_fun', ('temp', 1), f, [('param', 0)])` which means `temp[1] = f(param[0])`.
        - `('if_else', ('temp', 0), 5)` which means `if temp[0] != 0 goto label 5`.
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
        ('mul', ('out', 0), [('param', 0), ('param', 0)])
        ('fun', ('temp', 1), cos, ('param', 0))
        ('add', ('out', 0), [('const', 0), ('out', 0), ('temp', 1)])
        temp list length: 2
        constants: [5/3]
        ```
        """

    def merge(self, other: Evaluator, cpe_rounds: Optional[int] = None) -> Evaluator:
        """
        Merge evaluator `other` into `self`. The parameters must be the same, and
        the outputs will be concatenated.

        The optional `cpe_rounds` parameter can be used to limit the number of common
        pair elimination rounds after the merge.

        Examples
        --------

        >>> from symbolica import *
        >>> e1 = E('x').evaluator({}, {}, [S('x')])
        >>> e2 = E('x+1').evaluator({}, {}, [S('x')])
        >>> e = e1.merge(e2)
        >>> e.evaluate([2])

        yields `[2, 3]`.
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
        compiler_path: Optional[str] = None,
        compiler_flags: Optional[Sequence[str]] = None,
        custom_header: Optional[str] = None,
    ) -> CompiledRealEvaluator:
        """Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

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
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : Optional[str]
            The custom path to the compiler executable.
        compiler_flags : Optional[Sequence[str]]
            The custom flags to pass to the compiler.
        custom_header : Optional[str]
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
        compiler_path: Optional[str] = None,
        compiler_flags: Optional[Sequence[str]] = None,
        custom_header: Optional[str] = None,
    ) -> CompiledComplexEvaluator:
        """Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

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
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : Optional[str]
            The custom path to the compiler executable.
        compiler_flags : Optional[Sequence[str]]
            The custom flags to pass to the compiler.
        custom_header : Optional[str]
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
        compiler_path: Optional[str] = None,
        compiler_flags: Optional[Sequence[str]] = None,
        custom_header: Optional[str] = None,
    ) -> CompiledSimdRealEvaluator:
        """Compile the evaluator to a shared library with 4x SIMD using C++ and optionally inline assembly and load it.

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
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : Optional[str]
            The custom path to the compiler executable.
        compiler_flags : Optional[Sequence[str]]
            The custom flags to pass to the compiler.
        custom_header : Optional[str]
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
        compiler_path: Optional[str] = None,
        compiler_flags: Optional[Sequence[str]] = None,
        custom_header: Optional[str] = None,
    ) -> CompiledSimdComplexEvaluator:
        """Compile the evaluator to a shared library with 4x SIMD using C++ and optionally inline assembly and load it.

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
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
        compiler_path : Optional[str]
            The custom path to the compiler executable.
        compiler_flags : Optional[Sequence[str]]
            The custom flags to pass to the compiler.
        custom_header : Optional[str]
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
        compiler_path: Optional[str] = None,
        compiler_flags: Optional[Sequence[str]] = None,
        custom_header: Optional[str] = None,
        cuda_number_of_evaluations: Optional[int] = None,
        cuda_block_size: Optional[int] = 256
    ) -> CompiledCudaRealEvaluator:
        """Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

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
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
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
            The block size to use for CUDA kernel launches.
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
        compiler_path: Optional[str] = None,
        compiler_flags: Optional[Sequence[str]] = None,
        custom_header: Optional[str] = None,
        cuda_number_of_evaluations: Optional[int] = None,
        cuda_block_size: Optional[int] = 256
    ) -> CompiledCudaComplexEvaluator:
        """Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

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
            The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
            For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
            For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
        inline_asm : str
            The inline ASM option can be set to 'default', 'x64', 'avx2', 'aarch64' or 'none'.
        optimization_level : int
            The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
        native: bool
            If `True`, compile for the native architecture. This may produce faster code, but is less portable.
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
            The block size to use for CUDA kernel launches.
        """

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the expression for multiple inputs and return the result.
        For best performance, use `numpy` arrays instead of lists.

        Examples
        --------
        Evaluate the function for three sets of inputs:

        >>> from symbolica import *
        >>> import numpy as np
        >>> ev = E('x * y + 2').evaluator({}, {}, [S('x'), S('y')])
        >>> print(ev.evaluate(np.array([1., 2., 3., 4., 5., 6.]).reshape((3, 2))))

        Yields`[[ 4.] [ 8.] [14.]]` 
        """

    def evaluate_complex(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Evaluate the expression for multiple inputs and return the result.
        For best performance, use `numpy` arrays and `np.complex128` instead of lists and
        `complex`.

        Examples
        --------
        Evaluate the function for three sets of inputs:

        >>> from symbolica import *
        >>> import numpy as np
        >>> ev = E('x * y + 2').evaluator({}, {}, [S('x'), S('y')])
        >>> print(ev.evaluate(np.array([1.+2j, 2., 3., 4., 5., 6.]).reshape((3, 2))))

        Yields`[[ 4.+4.j] [14.+0.j] [32.+0.j]]` 
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
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the expression for multiple inputs and return the result."""


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
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Evaluate the expression for multiple inputs and return the result."""


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
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the expression for multiple inputs and return the result."""


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
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Evaluate the expression for multiple inputs and return the result."""


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
        cuda_block_size: Optional[int] = 256
    ) -> CompiledCudaRealEvaluator:
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the expression for multiple inputs and return the result."""


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
        cuda_block_size: Optional[int] = 256
    ) -> CompiledCudaComplexEvaluator:
        """Load a compiled library, previously generated with `Evaluator.compile()`."""

    def evaluate(self, inputs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Evaluate the expression for multiple inputs and return the result."""


class NumericalIntegrator:
    """A numerical integrator for high-dimensional integrals."""

    def __copy__(self) -> NumericalIntegrator:
        """Copy the grid without any unprocessed samples."""

    @classmethod
    def continuous(
        _cls,
        n_dims: int,
        n_bins: int = 128,
        min_samples_for_update: int = 100,
        bin_number_evolution: Optional[Sequence[int]] = None,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        """Create a new continuous grid for the numerical integrator."""

    @classmethod
    def discrete(
        _cls,
        bins: Sequence[Optional[NumericalIntegrator]],
        max_prob_ratio: float = 100.0,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        """Create a new discrete grid for the numerical integrator. Each
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
        """

    @classmethod
    def uniform(
        _cls,
        bins: Sequence[int],
        continuous_subgrid: NumericalIntegrator,
    ) -> NumericalIntegrator:
        """Create a new uniform layered grid for the numerical integrator.
        `len(bins)` specifies the number of discrete layers, and each entry in `bins` specifies the number of bins in that layer.
        Each discrete bin has equal probability.

        Examples
        --------
        >>> def integrand(samples: typing.Sequence[Sample]) -> list[float]:
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
        """

    @classmethod
    def rng(
        _cls,
        seed: int,
        stream_id: int
    ) -> RandomNumberGenerator:
        """Create a new random number generator, suitable for use with the integrator.
        Each thread of instance of the integrator should have its own random number generator,
        that is initialized with the same seed but with a different stream id."""

    @classmethod
    def import_grid(
        _cls,
        grid: bytes
    ) -> NumericalIntegrator:
        """Import an exported grid from another thread or machine.
        Use `export_grid` to export the grid."""

    def export_grid(
        self,
        export_samples: bool = True,
    ) -> bytes:
        """Export the grid, so that it can be sent to another thread or machine.
        If you are exporting your main grid, make sure to set `export_samples` to `False` to avoid copying unprocessed samples.

        Use `import_grid` to load the grid.        
        """

    def get_live_estimate(
        self,
    ) -> Tuple[float, float, float, float, float, int]:
        """Get the estamate of the average, error, chi-squared, maximum negative and positive evaluations, and the number of processed samples
        for the current iteration, including the points submitted in the current iteration."""

    def sample(self, num_samples: int, rng: RandomNumberGenerator) -> List[Sample]:
        """Sample `num_samples` points from the grid using the random number generator
        `rng`. See `rng()` for how to create a random number generator."""

    def merge(self, other: NumericalIntegrator) -> None:
        """Add the accumulated training samples from the grid `other` to the current grid.
        The grid structure of `self` and `other` must be equivalent."""

    def add_training_samples(self, samples: Sequence[Sample], evals: Sequence[float]) -> None:
        """Add the samples and their corresponding function evaluations to the grid.
        Call `update` after to update the grid and to obtain the new expected value for the integral."""

    def update(self, discrete_learning_rate: float, continous_learning_rate: float) -> Tuple[float, float, float]:
        """Update the grid using the `discrete_learning_rate` and `continuous_learning_rate`.
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
        """

    def integrate(
        self,
        integrand: Callable[[Sequence[Sample]], List[float]],
        max_n_iter: int = 10000000,
        min_error: float = 0.01,
        n_samples_per_iter: int = 10000,
        seed: int = 0,
        show_stats: bool = True,
    ) -> Tuple[float, float, float]:
        """Integrate the function `integrand` that maps a list of `Sample`s to a list of `float`s.
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
        """


class Sample:
    """A sample from the Symbolica integrator. It could consist of discrete layers,
    accessible with `d` (empty when there are not discrete layers), and the final continuous layer `c` if it is present."""

    """ The weights the integrator assigned to this sample point, given in descending order:
    first the discrete layer weights and then the continuous layer weight."""
    weights: List[float]
    d: List[int]
    """ A sample point per (nested) discrete layer. Empty if not present."""
    c: List[float]
    """ A sample in the continuous layer. Empty if not present."""


class RandomNumberGenerator:
    """A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
    A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).

    Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
    which is an instance counter starting at 0."""

    def __new__(_cls, seed: int, stream_id: int):
        """Create a new random number generator with a given `seed` and `stream_id`. For parallel runs,
        each thread or instance generating samples should use the same `seed` but a different `stream_id`."""


class HalfEdge:
    """A half-edge in a graph that connects to one vertex, consisting of a direction (or `None` if undirected) and edge data."""

    def __new__(_cls, data: Expression | int, direction: Optional[bool] = None):
        """Create a new half-edge. The `data` can be any expression, and the `direction` can be `True` (outgoing),
        `False` (incoming) or `None` (undirected).
        """

    def flip(self) -> HalfEdge:
        """Return a new half-edge with the direction flipped (if it has a direction)."""

    def direction(self) -> Optional[bool]:
        """Get the direction of the half-edge. `True` means outgoing, `False` means incoming, and `None` means undirected."""

    def data(self) -> Expression:
        """Get the data of the half-edge."""


class Graph:
    """A graph that supported directional edges, parallel edges, self-edges and expression data on the nodes and edges.

    Warning: modifying the graph if it is contained in a `dict` or `set` will invalidate the hash.
    """

    def __new__(_cls):
        """Create a new empty graph."""

    def __str__(self) -> str:
        """Print the graph in a human-readable format."""

    def __hash__(self) -> int:
        """Hash the graph."""

    def __copy__(self) -> Graph:
        """Copy the graph."""

    def __len__(self) -> int:
        """Get the number of nodes in the graph."""

    def __eq__(self, other: Graph) -> bool:
        """Compare two graphs."""

    def __neq__(self, other: Graph) -> bool:
        """Compare two graphs."""

    def __getitem__(self, idx: int) -> tuple[Sequence[int], Expression]:
        """Get the `idx`th node, consisting of the edge indices and the data."""

    @classmethod
    def generate(_cls,
                 external_nodes: Sequence[tuple[Expression | int, HalfEdge]],
                 vertex_signatures: Sequence[Sequence[HalfEdge]],
                 max_vertices: Optional[int] = None,
                 max_loops: Optional[int] = None,
                 max_bridges: Optional[int] = None,
                 allow_self_loops: bool = False,
                 allow_zero_flow_edges: bool = False,
                 filter_fn: Optional[Callable[[Graph, int], bool]] = None,
                 progress_fn: Optional[Callable[[Graph], bool]] = None) -> dict[Graph, Expression]:
        """Generate all connected graphs with `external_edges` half-edges and the given allowed list
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
        filter_fn: Optional[Callable[[Graph, int], bool]], optional
            Set a filter function that is called during the graph generation.
            The first argument is the graph `g` and the second argument the vertex count `n`
            that specifies that the first `n` vertices are completed (no new edges will) be
            assigned to them. The filter function should return `true` if the current
            incomplete graph is allowed, else it should return `false` and the graph is discarded.
        progress_fn: Optional[Callable[[Graph], bool]], optional
            Set a progress function that is called every time a new unique graph is created.
            The argument is the currently generated graph.
            If the function returns `false`, the generation is aborted and the currently
            generated graphs are returned.
        """

    def to_dot(self) -> str:
        """Convert the graph to a graphviz dot string."""

    def to_mermaid(self) -> str:
        """Convert the graph to a mermaid string."""

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""

    def num_edges(self) -> int:
        """Get the number of edges in the graph."""

    def num_loops(self) -> int:
        """Get the number of loops in the graph."""

    def node(self, idx: int) -> tuple[Sequence[int], Expression]:
        """Get the `idx`th node, consisting of the edge indices and the data."""

    def nodes(self) -> list[tuple[Sequence[int], Expression]]:
        """Get all nodes, consisting of the edge indices and the data."""

    def edge(self, idx: int) -> tuple[int, int, bool, Expression]:
        """Get the `idx`th edge, consisting of the the source vertex, target vertex, whether the edge is directed, and the data."""

    def edges(self) -> list[tuple[int, int, bool, Expression]]:
        """Get all edges, consisting of the the source vertex, target vertex, whether the edge is directed, and the data."""

    def add_node(self, data: Optional[Expression | int] = None) -> int:
        """Add a node with data `data` to the graph, returning the index of the node.
        The default data is the number 0.
        """

    def add_edge(self, source: int, target: int, directed: bool = False, data: Optional[Expression | int] = None) -> int:
        """Add an edge between the `source` and `target` nodes, returning the index of the edge.

        Optionally, the edge can be set as directed. The default data is the number 0.
        """

    def set_node_data(self, index: int, data: Expression | int) -> Expression:
        """Set the data of the node at index `index`, returning the old data."""

    def set_edge_data(self, index: int, data: Expression | int) -> Expression:
        """Set the data of the edge at index `index`, returning the old data."""

    def set_directed(self, index: int, directed: bool) -> bool:
        """Set the directed status of the edge at index `index`, returning the old value."""

    def canonize(self) -> Tuple[Graph, Sequence[int], Expression, Sequence[int]]:
        """Write the graph in a canonical form. Returns the canonized graph, the vertex map, the automorphism group size, and the orbit."""

    def canonize_edges(self) -> None:
        """Sort and relabel the edges of the graph, keeping the vertices fixed."""

    def is_isomorphic(self, other: Graph) -> bool:
        """Check if the graph is isomorphic to another graph."""


class Integer:
    @classmethod
    def prime_iter(_cls, start: int = 1) -> Iterator[int]:
        """Create an iterator over all 64-bit prime numbers starting from `start`."""

    @classmethod
    def is_prime(_cls, n: int) -> bool:
        """Check if the 64-bit number `n` is a prime number."""

    @classmethod
    def factor(_cls, n: int) -> Sequence[Tuple[int, int]]:
        """Factor the 64-bit number `n` into its prime factors and return a list of tuples `(p, e)` where `p` is a prime factor and `e` is its exponent."""

    @classmethod
    def totient(_cls, n: int) -> int:
        """Compute the Euler totient function of the number `n`, i.e., the number of integers less than `n` that are coprime to `n`."""

    @classmethod
    def gcd(_cls, a: int, b: int) -> int:
        """Compute the greatest common divisor of the numbers `a` and `b`."""

    @classmethod
    def lcm(_cls, a: int, b: int) -> int:
        """Compute the least common multiple of the numbers `a` and `b`."""

    @classmethod
    def extended_gcd(_cls, a: int, b: int) -> tuple[int, int, int]:
        """Compute the greatest common divisor of the numbers `a` and `b` and the Bézout coefficients."""

    @classmethod
    def chinese_remainder(_cls, n1: int, m1: int, n2: int, m2: int) -> int:
        """
        Solve the Chinese remainder theorem for the equations:
        `x = n1 mod m1` and `x = n2 mod m2`.
        """

    @classmethod
    def solve_integer_relation(_cls, x: Sequence[int | float | complex | Decimal], tolerance: float | Decimal, max_coeff: Optional[int] = None, gamma: Optional[float | Decimal] = None) -> Sequence[int]:
        """Use the PSLQ algorithm to find a vector of integers `a` that satisfies `a.x = 0`,
        where every element of `a` is less than `max_coeff`, using a specified tolerance and number
        of iterations. The parameter `gamma` must be more than or equal to `2/sqrt(3)`.

        Examples
        --------
        Solve a `32.0177=b*pi+c*e` where `b` and `c` are integers:

        >>> r = Integer.solve_integer_relation([-32.0177, 3.1416, 2.7183], 1e-5, 100)
        >>> print(r)

        yields `[1,5,6]`.
        """
