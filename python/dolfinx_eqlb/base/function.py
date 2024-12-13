from dataclasses import dataclass
from mpi4py import MPI
import numpy as np
from numpy.typing import DTypeLike, NDArray
import typing

import ffcx
from dolfinx import default_scalar_type, jit, fem
from dolfinx import cpp as _cpp
import ufl


@dataclass
class CompiledExpression:
    """Compiled UFL expression without associated DOLFINx data."""

    ufl_expression: ufl.core.expr.Expr  # The original ufl form
    ufcx_form: typing.Any  # The compiled form
    module: typing.Any  #  The module
    code: str  # The source code
    dtype: DTypeLike  # data type used for the `ufcx_form`


class Expression(fem.Expression):
    def __init__(
        self,
        e: CompiledExpression,
        constant_map: dict[ufl.Constant, fem.Constant],
        coefficient_map: dict[ufl.Coefficient, fem.Function],
    ):
        """Create a DOLFINx Expression.

        Represents a mathematical expression evaluated at a pre-defined
        set of points on the reference cell.

        This definition follows straight-forwardly the definition in DOLFINx,
        but cam be created from a pre-compiled abstract expression.

        Args:
            e: A pre-compiled abstract expression.
            constant_map: Map from UFL constant to constant with data.
            coefficient_map: Map from UFL coefficient to function with data.
        """

        # Extract from precompiled expression
        self._ufcx_expression = e.ufcx_form
        self._code = e.code
        self._ufl_expression = e.ufl_expression

        # Prepare constants and coefficients
        constants, coefficients = extract_constants_and_coefficients(
            self.ufl_expression, self._ufcx_expression, constant_map, coefficient_map
        )

        # Get related function space
        arguments = ufl.algorithms.extract_arguments(self._ufl_expression)
        if len(arguments) == 0:
            self._argument_function_space = None
        elif len(arguments) == 1:
            self._argument_function_space = (
                arguments[0].ufl_function_space()._cpp_object
            )
        else:
            raise RuntimeError("Expressions with more that one Argument not allowed.")

        def _create_expression(dtype):
            if np.issubdtype(dtype, np.float32):
                return _cpp.fem.create_expression_float32
            elif np.issubdtype(dtype, np.float64):
                return _cpp.fem.create_expression_float64
            elif np.issubdtype(dtype, np.complex64):
                return _cpp.fem.create_expression_complex64
            elif np.issubdtype(dtype, np.complex128):
                return _cpp.fem.create_expression_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        self._cpp_object = _create_expression(e.dtype)(
            e.module.ffi.cast(
                "uintptr_t", e.module.ffi.addressof(self._ufcx_expression)
            ),
            coefficients,
            constants,
            self.argument_function_space,
        )


def extract_constants_and_coefficients(
    ufl_expression: ufl.core.expr.Expr,
    ufcx_expression: typing.Any,
    constant_map: typing.Dict[ufl.Constant, fem.Constant],
    coefficient_map: typing.Dict[ufl.Coefficient, fem.Function],
) -> typing.Tuple[typing.List[ufl.Constant], typing.List[ufl.Coefficient]]:
    """Prepare mesh-dependent data of a pre-compiled UFL expression.

    Args:
        ufl_expression:  The UFL expression
        ufcx_expression: The compiled UFL expression
        constant_map:    Map from UFL constant to constant with data
        coefficient_map: Map from UFL coefficient to function with data
    """

    ufl_coefficients = ufl.algorithms.extract_coefficients(ufl_expression)
    coefficients = []

    for i in range(ufcx_expression.num_coefficients):
        # Get the ufl coefficient
        ufl_coeff_i = ufl_coefficients[
            ufcx_expression.original_coefficient_positions[i]
        ]

        # Replace it by the corresponding function
        coefficients.append(coefficient_map[ufl_coeff_i]._cpp_object)

    ufl_constants = ufl.algorithms.analysis.extract_constants(ufl_expression)
    constants = [constant_map[ufl_const]._cpp_object for ufl_const in ufl_constants]

    return constants, coefficients


def compile_expression(
    comm: MPI.Intracomm,
    expr: ufl.core.expr.Expr,
    pnts: NDArray[np.float64],
    form_compiler_options: typing.Optional[dict] = {"scalar_type": default_scalar_type},
    jit_options: typing.Optional[dict] = None,
) -> CompiledExpression:
    """Compile UFL expression without associated DOLFINx data.

    Args:
        comm: The MPI communicator used when compiling the form
        expr: The UFL expression to compile
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
    """

    p_ffcx = ffcx.get_options(form_compiler_options)
    p_jit = jit.get_options(jit_options)
    ufcx_form, module, code = jit.ffcx_jit(comm, (expr, pnts), p_ffcx, p_jit)

    return CompiledExpression(expr, ufcx_form, module, code, p_ffcx["scalar_type"])
