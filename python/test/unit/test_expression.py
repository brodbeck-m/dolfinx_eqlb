from mpi4py import MPI
import numpy as np
import pytest

import basix
from basix import ufl
from dolfinx import fem, mesh
import ufl

from dolfinx_eqlb.base import compile_expression, Expression


# --- Auxiliaries ---
def compute_exterior_facet_entities(domain, facets):
    """Helper function to compute (cell, local_facet_index) pairs for exterior facets"""
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    domain.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = domain.topology.connectivity(tdim, tdim - 1)
    f_to_c = domain.topology.connectivity(tdim - 1, tdim)
    integration_entities = np.empty(2 * len(facets), dtype=np.int32)
    for i, facet in enumerate(facets):
        cells = f_to_c.links(facet)
        assert len(cells) == 1
        cell = cells[0]
        local_facets = c_to_f.links(cell)
        local_pos = np.flatnonzero(local_facets == facet)
        integration_entities[2 * i] = cell
        integration_entities[2 * i + 1] = local_pos[0]

    return integration_entities


# --- The tests ---
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rank0(dtype):
    """Test evaluation of UFL expression.

    This test evaluates gradient of P2 function at interpolation points
    of vector dP1 element.

    For a donor function cf * f(x, y) + cg * g(x, y), where cf and cg are
    two constants, the result is compared with the exact gradient.
    """

    # --- Mesh-independent definitions
    # The abstracted mesh
    c_el = basix.ufl.element(
        "Lagrange", "triangle", 1, shape=(2,), dtype=dtype(0).real.dtype
    )
    domain = ufl.Mesh(c_el)

    # The data
    element_u = basix.ufl.element("Lagrange", "triangle", 2, dtype=dtype(0).real.dtype)
    element_du = basix.ufl.element(
        "DG", "triangle", 1, shape=(2,), dtype=dtype(0).real.dtype
    )
    V = ufl.FunctionSpace(domain, element_u)

    f = ufl.Coefficient(V)
    cf = ufl.Constant(domain)

    g = ufl.Coefficient(V)
    cg = ufl.Constant(domain)

    # The expression
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    compiled_expr = compile_expression(
        MPI.COMM_WORLD,
        cf * ufl.grad(f) + cg * ufl.grad(g),
        points,
        {"scalar_type": dtype(0).real.dtype},
    )

    # Check expression on a series of meshes
    def evaluate_gradient(n, complied_expr):
        #  The mesh
        domain_h = mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, dtype=dtype(0).real.dtype
        )

        # The data
        P2 = fem.functionspace(domain_h, element_u)
        vdP1 = fem.functionspace(domain_h, element_du)

        f_h = fem.Function(P2, dtype=dtype)
        f_h.interpolate(lambda x: x[0] ** 2 + 2.0 * x[1] ** 2)
        cf_h = fem.Constant(domain_h, dtype(2.5))

        g_h = fem.Function(P2, dtype=dtype)
        g_h.interpolate(lambda x: 0.5 * x[0] ** 2 + x[1] ** 2)
        cg_h = fem.Constant(domain_h, dtype(1.5))

        # Expression with data
        expr_h = Expression(complied_expr, {cf: cf_h, cg: cg_h}, {f: f_h, g: g_h})
        num_cells = domain_h.topology.index_map(2).size_local
        array_evaluated = expr_h.eval(domain_h, np.arange(num_cells, dtype=np.int32))

        def scatter(vec, array_evaluated, dofmap):
            for i in range(num_cells):
                for j in range(3):
                    for k in range(2):
                        vec[2 * dofmap[i * 3 + j] + k] = array_evaluated[i, 2 * j + k]

        # Data structure for the result
        b = fem.Function(vdP1, dtype=dtype)
        dofmap = vdP1.dofmap.list.flatten()
        scatter(b.x.array, array_evaluated, dofmap)
        b.x.scatter_forward()

        b2 = fem.Function(vdP1, dtype=dtype)
        b2.interpolate(lambda x: np.vstack((6.5 * x[0], 13 * x[1])))

        assert np.allclose(
            b2.x.array,
            b.x.array,
            rtol=np.sqrt(np.finfo(dtype).eps),
            atol=np.sqrt(np.finfo(dtype).eps),
        )

    for n in range(1, 4):
        evaluate_gradient(n, compiled_expr)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_facet_expression(dtype):
    xtype = dtype(0).real.dtype

    # --- Mesh-independent definitions
    # The abstracted mesh
    c_el = basix.ufl.element(
        "Lagrange", "triangle", 1, shape=(2,), dtype=dtype(0).real.dtype
    )
    domain = ufl.Mesh(c_el)

    # The expression
    reference_midpoint, _ = basix.quadrature.make_quadrature(
        basix.cell.CellType.interval,
        1,
        basix.quadrature.QuadratureType.default,
        basix.quadrature.PolysetType.standard,
    )

    compiled_expr = compile_expression(
        MPI.COMM_WORLD,
        ufl.FacetNormal(domain),
        reference_midpoint,
        {"scalar_type": xtype},
    )

    # --- Check expression on a series of meshes
    def evaluate_normal(n, complied_expr):
        #  The mesh
        domain_h = mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, dtype=dtype(0).real.dtype
        )

        # The boundary facets
        tdim = domain_h.topology.dim
        domain_h.topology.create_connectivity(tdim - 1, tdim)
        facets = mesh.exterior_facet_indices(domain_h.topology)
        boundary_entities = compute_exterior_facet_entities(domain_h, facets)

        # Expression with data
        expr_h = Expression(complied_expr, {}, {})
        facet_normals = expr_h.eval(domain_h, boundary_entities)

        # Check facet normal by using midpoint to determine what exterior cell we are at
        facet_midpoints = mesh.compute_midpoints(domain_h, tdim - 1, facets)
        atol = 100 * np.finfo(dtype).resolution
        for midpoint, normal in zip(facet_midpoints, facet_normals):
            if np.isclose(midpoint[0], 0, atol=atol):
                assert np.allclose(normal, [-1, 0])
            elif np.isclose(midpoint[0], 1, atol=atol):
                assert np.allclose(normal, [1, 0], atol=atol)
            elif np.isclose(midpoint[1], 0):
                assert np.allclose(normal, [0, -1], atol=atol)
            elif np.isclose(midpoint[1], 1, atol=atol):
                assert np.allclose(normal, [0, 1])
            else:
                raise ValueError("Invalid midpoint")

        # Check expression with coefficients from mixed space
        el_v = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,), dtype=xtype)
        el_p = basix.ufl.element("Lagrange", "triangle", 1, dtype=xtype)
        mixed_el = basix.ufl.mixed_element([el_v, el_p])
        W = fem.functionspace(domain_h, mixed_el)
        w = fem.Function(W, dtype=dtype)
        w.sub(0).interpolate(
            lambda x: (x[1] ** 2 + 3 * x[0] ** 2, -5 * x[1] ** 2 - 7 * x[0] ** 2)
        )
        w.sub(1).interpolate(lambda x: 2 * (x[1] + x[0]))
        u, p = ufl.split(w)
        n = ufl.FacetNormal(domain_h)
        mixed_expr = p * ufl.dot(ufl.grad(u), n)
        facet_expression = fem.Expression(
            mixed_expr, np.array([[0.5]], dtype=dtype), dtype=dtype
        )
        subset_values = facet_expression.eval(domain_h, boundary_entities)
        for values, midpoint in zip(subset_values, facet_midpoints):
            grad_u = np.array(
                [
                    [6 * midpoint[0], 2 * midpoint[1]],
                    [-14 * midpoint[0], -10 * midpoint[1]],
                ],
                dtype=dtype,
            )
            if np.isclose(midpoint[0], 0, atol=atol):
                exact_n = [-1, 0]
            elif np.isclose(midpoint[0], 1, atol=atol):
                exact_n = [1, 0]
            elif np.isclose(midpoint[1], 0):
                exact_n = [0, -1]
            elif np.isclose(midpoint[1], 1, atol=atol):
                exact_n = [0, 1]

            exact_expr = 2 * (midpoint[1] + midpoint[0]) * np.dot(grad_u, exact_n)
            assert np.allclose(values, exact_expr, atol=atol)

    for n in range(1, 4):
        evaluate_normal(n, compiled_expr)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
