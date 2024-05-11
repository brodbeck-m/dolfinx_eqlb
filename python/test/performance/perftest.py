# --- Imports ---
from enum import Enum
from mpi4py import MPI
import numpy as np
import time

import dolfinx.mesh as dmesh

from perftest_basics import (
    Testcases,
    SolverType,
    setup_testcase,
    projection,
    assemble_primal_problem,
)

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

# --- Parameters ---
# The testcase
testcase = Testcases.Poisson

# The orders of the FE spaces
orders = [1, 2, 3, 4]

# The solver (linear equation system)
solver_type = SolverType.cg_amg

# Initial dicretisation
n_elmts_init = 10

# Timing options
n_refinements = 5
n_repeats = 3


# --- Performance test ---
class Results(Enum):
    ncells = 0
    nnodes = 1
    tp_assembly = 2
    tp_solve = 3
    tp_total = 4
    t_projection = 5
    t_eqlb_SE = 6
    t_eqlb_EV = 7


# --- Initialisations
# The storage
timings = np.zeros((n_refinements, len(Results), len(orders)))
timings_max = np.zeros((n_refinements, len(Results), len(orders)))
minmaxratio = np.zeros((n_refinements, len(Results), len(orders)))

# The mesh
ctype = dmesh.CellType.triangle
dtype = dmesh.DiagonalType.crossed

# --- Time all cases
for i in range(n_refinements):
    # Number of elements
    n_elmt = n_elmts_init * (2**i)

    # Create mesh
    domain = dmesh.create_unit_square(
        MPI.COMM_WORLD, n_elmt, n_elmt, cell_type=ctype, diagonal=dtype
    )

    timings[i, Results.ncells.value, :] = domain.topology.index_map(2).size_local
    timings[i, Results.nnodes.value, :] = domain.topology.index_map(0).size_local

    for j, order in enumerate(orders):
        # Setup primal problem
        (a_prime, l_prime, uh, a_proj, l_proj, res_proj, rhs_eqlb, bfcts, bcs_esnt) = (
            setup_testcase(testcase, domain, order, order)
        )

        # Setup equilibrators
        equilibrators = []

        if testcase == Testcases.Poisson:
            equilibrators.append(FluxEqlbSE(order, domain, rhs_eqlb, [res_proj[1]]))
            equilibrators[-1].set_boundary_conditions([bfcts], [[]])

            equilibrators.append(FluxEqlbEV(order, domain, rhs_eqlb, [res_proj[1]]))
            equilibrators[-1].set_boundary_conditions([bfcts], [[]])
        elif testcase == Testcases.Elasticity:
            equilibrators.append(
                FluxEqlbSE(
                    order,
                    domain,
                    rhs_eqlb,
                    [res_proj[1], res_proj[2]],
                    equilibrate_stress=True,
                )
            )
            equilibrators[-1].set_boundary_conditions([bfcts, bfcts], [[], []])
        elif testcase == Testcases.Biot_upp:
            equilibrators.append(
                FluxEqlbSE(
                    order,
                    domain,
                    rhs_eqlb,
                    [res_proj[2], res_proj[3], res_proj[4]],
                    equilibrate_stress=True,
                )
            )
            equilibrators[-1].set_boundary_conditions(
                [bfcts, bfcts, bfcts], [[], [], []]
            )

        # Time problem
        for r in range(n_repeats):
            print(
                "Refinement {} of {}; Order {} of {}; Repetition {} or {}".format(
                    i + 1, n_refinements, j + 1, len(orders), r + 1, n_repeats
                )
            )

            # Assemble primal problem
            tp_assemble = -time.perf_counter()
            solver, L = assemble_primal_problem(a_prime, l_prime, bcs_esnt, solver_type)
            tp_assemble += time.perf_counter()

            # Solve primal problem
            tp_solve = -time.perf_counter()
            solver(L, uh.vector)
            uh.x.scatter_forward()
            tp_solve += time.perf_counter()

            # Required projections
            t_projection = -time.perf_counter()
            projection(a_proj, l_proj, res_proj, rhs_eqlb)
            t_projection += time.perf_counter()

            # Setup equilibrators
            t_eqlb = []
            for eqlb in equilibrators:
                te = -time.perf_counter()
                eqlb.equilibrate_fluxes()
                te += time.perf_counter()

                t_eqlb.append(te)

            # Store timings
            if r == 0:
                timings[i, Results.tp_assembly.value, j] = tp_assemble
                timings[i, Results.tp_solve.value, j] = tp_solve
                timings[i, Results.t_projection.value, j] = t_projection
                for k, te in enumerate(t_eqlb):
                    timings[i, Results.t_eqlb_SE.value + k, j] = te

                timings_max[i, Results.tp_assembly.value, j] = tp_assemble
                timings_max[i, Results.tp_solve.value, j] = tp_solve
                timings_max[i, Results.t_projection.value, j] = t_projection
                for k, te in enumerate(t_eqlb):
                    timings_max[i, Results.t_eqlb_SE.value + k, j] = te
            else:
                if timings[i, Results.tp_assembly.value, j] > tp_assemble:
                    timings[i, Results.tp_assembly.value, j] = tp_assemble
                if timings_max[i, Results.tp_assembly.value, j] < tp_assemble:
                    timings_max[i, Results.tp_assembly.value, j] = tp_assemble

                if timings[i, Results.tp_solve.value, j] > tp_solve:
                    timings[i, Results.tp_solve.value, j] = tp_solve
                if timings_max[i, Results.tp_solve.value, j] < tp_solve:
                    timings_max[i, Results.tp_solve.value, j] = tp_solve

                if timings[i, Results.t_projection.value, j] > t_projection:
                    timings[i, Results.t_projection.value, j] = t_projection
                if timings_max[i, Results.t_projection.value, j] < t_projection:
                    timings_max[i, Results.t_projection.value, j] = t_projection

                for k, te in enumerate(t_eqlb):
                    if timings[i, Results.t_eqlb_SE.value + k, j] > te:
                        timings[i, Results.t_eqlb_SE.value + k, j] = te
                    if timings_max[i, Results.t_eqlb_SE.value + k, j] < te:
                        timings_max[i, Results.t_eqlb_SE.value + k, j] = te

# --- Output results
basename = "perftest"
header_protocol = (
    "nelmt, nnodes, tpassembly, tpsolve, tptotal, tprojection, teqlbSE, teqlbEV"
)

if testcase == Testcases.Poisson:
    basename += "_poisson"
elif testcase == Testcases.Elasticity:
    basename += "_elasticity"
elif testcase == Testcases.Biot_up:
    basename += "_biot-up"
elif testcase == Testcases.Biot_upp:
    basename += "_biot-upp"

for k, order in enumerate(orders):
    # Total solution time primal problem
    timings[:, Results.tp_total.value, k] = (
        timings[:, Results.tp_assembly.value, k] + timings[:, Results.tp_solve.value, k]
    )
    timings_max[:, Results.tp_total.value, k] = (
        timings_max[:, Results.tp_assembly.value, k]
        + timings_max[:, Results.tp_solve.value, k]
    )

    # Min-to-Max ratio
    minmaxratio[:, 0:2, k] = timings[:, 0:2, k]
    minmaxratio[:, 2:, k] = np.divide(timings_max[:, 2:, k], timings[:, 2:, k])

    # Output results
    np.savetxt(
        basename + "_order-" + str(order) + ".csv",
        timings[:, :, k],
        delimiter=",",
        header=header_protocol,
    )

    np.savetxt(
        basename + "-minmaxratio" + "_order-" + str(order) + ".csv",
        minmaxratio[:, :, k],
        delimiter=",",
        header=header_protocol,
    )
