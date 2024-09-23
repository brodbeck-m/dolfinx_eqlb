# dolfinx for flux equilibration (dolfinx_eqlb)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--4459-d45815.svg)](https://doi.org/10.18419/darus-4459)

Author: Maximilian Brodbeck

This library contains an add-on to FEniCSx enabling local flux equilibration strategies. The resulting H(div) conforming fluxes can be used for the construction of adaptive finite element solvers for the Poisson problem [[1]](#1), elasticity [[2]](#2)[[3]](#3)[[4]](#4) or poro-elasticity [[5]](#5)[[6]](#6).  

The equilibration process relies on so called patches, groups of all cells, connected with one node of the mesh. On each patch a constrained minimisation problem is solved [[7]](#7). In order to improve computational efficiency, a so called semi-explicit strategy [[8]](#8)[[9]](#9) is also implemented. The solution procedure is thereby split into two steps: An explicit determination of an H(div) function, fulfilling the minimisation constraints, followed by an unconstrained minimisation on a reduced, patch-wise ansatz space. If equilibration is applied to elasticity -- the stress tensor has a distinct symmetry -- an additional constrained minimisation step after the row wise reconstruction of the tensor [[3]](#3)[[4]](#4) is implemented.

### Features
dolfinx_eqlb supports flux equilibration on two-dimensional domains with arbitrary triangular grids. It further includes the following features
- A local projector into arbitrary function-spaces with cell-wise support
- A hierarchic Raviart-Thomas element based on Boffi, Brezzi and Fortin [[10]](#10)
- Boundary conditions for H(div) spaces on general boundaries
- Flux equilibration based on Ern and Vohralik (FluxEqlbEV) or a semi-explicit strategy (FluxEqlbSE)
- Stress equilibration considering distinct symmetry properties in a weak sense

### Getting started
1. Clone this repository using the command:

```shell
git clone git@github.com:brodbeck-m/dolfinx_eqlb.git
```

2. Download the required Docker image of DOLFINx:

```shell
docker pull dolfinx/dolfinx:v0.6.0-r1
```

3. Build a Docker image containing dolfinx_eqlb

```shell
cd docker
./build_image.sh 
```

4. Try out the basic demos. If no errors are reported, the equilibration process works as expected.

```shell
./launch-container.sh

# Equilibration for a Poisson problem
cd ./root/dolfinx_eqlb/python/demo/poisson
python3 demo_reconstruction.py  

# Equilibration for linear elasticity
cd ./root/dolfinx_eqlb/python/demo/elasticity
python3 demo_reconstruction.py  
```

## Documentation
Flux equilibration can either be used to improve the accuracy of dual quantity - e.g. the flux for a Poisson or heat equation or the stress in elasticity - by a post-processing step or as a basis for a-posteriori error estimation. Incorporating equilibration into a solution procedure required the following four steps:
1. Solve the primal problem based on Lagrangian finite elements of degree $k$.
2. Calculate the approximated dual quantity from the primal field.
3. Project the right-hand-side (RHS) of the primal problem as well as the approximated dual quantity into discontinuous Lagrange spaces of order $m \geq k$. The projection can be solved using a [local solver](#section_local-solver).
4. Equilibrate the dual quantity in an Raviart-Thomas space of order $m$.

The algorithmic structure of the equilibration itself is described in [AddSource]. A short description of the relevant Python interfaces of the library is given below. Complete examples for the Poisson equation and linear elasticity can be found in the [demo section](https://github.com/brodbeck-m/dolfinx_eqlb/tree/main/python/demo). 

### <a name="section_local-solver"></a> Local solvers
Projecting an arbitrary function $\mathrm{f}$ into a discontinuous finite element space $\mathrm{V}$ requires the solution of
$$ \left(\mathrm{u},\;\mathrm{v}\right) = \left(\mathrm{f},\;\mathrm{v}\right) $$
for all $\mathrm{v}\in\mathrm{V}$. As the function space $\mathrm{V}$ is discontinuous, the solution on each finite element can be computed independently. Assuming 'f_ufl' to be the ufl-representation of a function, the following code snippet shows the local projection:
```python
from dolfinx.fem import FunctionSpace
from dolfinx_eqlb.lsolver import local_projection

# Initialise target function space
V_proj = FunctionSpace(domain, ("DG", m - 1))

# Project f_ufl into fe-function
f_proj = local_projection(V_proj, [f_ufl])
```
```f_proj``` will be a list of functions, with the same length as the second argument of 'local_projection'. This allows the simultaneous projection of multiple function (as long as they have the same target function space), which is beneficial from a performance perspective, as the system matrix has to be factorised only once. Due to the symmetric and positive definite system matrix, a Cholesky decomposition is used.

### <a name="section_equilibrator"></a>Setting up the equilibrator
Based on projections of the approximated dual quantity and the RHS the equilibrator itself can be initialised. In order to improve efficiency, multiple RHS can be equilibrated at the same time. Assuming that for each $\bm{\varsigma}^\mathrm{R}_h$ a divergence condition of the form
$$\nabla\cdot\bm{\varsigma}^\mathrm{R}_h = \Pi_{m-1}\mathrm{f}$$
holds. $\Pi_{m-1}\left(\bullet\right)$ denotes the projection into a discontinous Lagrange space of order $m-1$. 

Having lists of DOLFINx functions with the projected RHS ```list_rhs```$=\left\{\mathrm{f}_i\right\}$
and the projected dual quantities ```list_sigmah```$=\left\{\bm{\varsigma}\left(\mathrm{u}_h\right)\big\vert_i\right\}$ the equilibrator is initialised as follows:
```python
from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

# Initialise equilibrated (approach by Ern and Vohralik [7])
equilibrator = FluxEqlbEV (m, domain , list_rhs , list_sigmah)

# Initialise equilibrated (semi-explicit approach [8,9])
equilibrator = FluxEqlbSE (m, domain , list_rhs , list_sigmah)
```
The semi-explicit equilibrator can be initialised with two optional arguments ```equilibrate_stress``` and ```estimate_korn_constant```. When the first one is set, the first ```gdim``` fluxes are treated as rows of a stress tensor and symmetry is enforced weakly [[3]](#3). The one enables the evaluation of the cells Korn constants (only for 2D) based on [[11]](#11). The Korn constants, stored within a $\mathrm{DP}_0$ function, can be extracted by the appropriate getter method:
```python
equilibrator.get_korn_constants()
```

Before the actual equilibration can be performed, boundary data namely the boundary facets on $\Gamma_\mathrm{D}$ and $\Gamma_\mathrm{N}$ as well as the normal traces of the flux on $\Gamma_\mathrm{N}$ , are required. While the facets lists are of type ```NDArray```, the (different) normal traces are stored in a list.

Assuming a domain with Dirichlet BCs on facets with ```facet_tags```$\in\left\{1,2\right\}$ and Neumann BCs for ```facet_tags```$\in\left\{3,4\right\}$, the following code snippet shows how to specify them:
```python
from numpy import isin
from dolfinx_eqlb.eqlb import fluxbc

# Get list of boundary facets
dftcs = facet_tags.indices[isin(facet_tags.values, [1, 2])]
nfcts = facet_tags.indices[isin(facet_tags.values, [3, 4])]

# Set Neumann boundary
bc = []
bc.append(fluxbc(f_neumann, nfcts, equilibrator.V_flux))
```
The function ```fluxbc``` has thereby the optional arguments ```requires_projection``` and ```quadrature_degree```. They are required when the traction lies not in the polynomial space of order $m-1$. Setting ```requires_projection``` enforces an $\mathrm{L}^2$-projection when the boundary DOFs are evaluated. The optionally specified ```quadrature_degree``` is then for the evaluation of the linear form of the projection. 

With these information provided for each simultaneously equilibrated flux the equilibration can be solved:
```python
# Set boundary conditions
equilibrator.set_boundary_conditions(list_dfcts, list_bcs)

# Solve equilibration
equilibrator.equilibrate_fluxes()
```

### Equilibrated fluxes for a-posteriori error estimation
Based on equilibrated fluxes reliable error estimates for different problem classes can be constructed. Showcases for the Poisson problem (estimate by Ern and Vohralik [[2]](#2)) and linear elasticity (following Bertrand et al. [[3]](#3)) are provided in the demo section. For both problems the equilibration- and error estimation process is demonstrated on a unit-square with manufactured solution on a series of uniformly refined meshes:
```shell
# Start the docker container
./docker/launch-container.sh

# --- Poisson problem ---
# Flux equilibration
cd ./root/dolfinx_eqlb/python/demo/poisson
python3 demo_reconstruction.py  

# Error estimation on a series of uniformly refined meshes
cd ./root/dolfinx_eqlb/python/demo/poisson
python3 demo_error_estimation.py  

# --- Linear elasticity ---
# Stress equilibration (with weak symmetry)
cd ./root/dolfinx_eqlb/python/demo/elasticity
python3 demo_reconstruction.py

# Error estimation on a series of uniformly refined meshes
cd ./root/dolfinx_eqlb/python/demo/elasticity
python3 demo_error_estimation.py  
```

Further examples on adaptively refined meshes are provided for [Poisson](https://github.com/brodbeck-m/dolfinx_eqlb/tree/main/python/demo/poisson_adaptive) and [linear elasticity](https://github.com/brodbeck-m/dolfinx_eqlb/tree/main/python/demo/elacticity_adaptive).

## How to cite
dolfinx_eqlb is a research software. The latest release can be found on [DaRUS](https://doi.org/10.18419/darus-4479), or - if citations of individual files or code lines are required - on [Software Heritage](???).

## Literature
<a id="1">[1]</a> Braess, D. and Schöberl, J.: Equilibrated Residual Error Estimator for Edge Elements (2008).

<a id="2">[2]</a> Prager, W. and Synge, J. L.: Approximations in elasticity based on the concept of function space (1947).

<a id="3">[3]</a> Bertrand et al.: Weakly symmetric stress equilibration and a posteriori error estimation for linear elasticity (2021).

<a id="4">[4]</a> Bertrand et al.: Weakly symmetric stress equilibration for hyperelastic material models (2020).

<a id="5">[5]</a> Riedlbeck et al.: Stress and flux reconstruction in Biot’s poro-elasticity problem with application to a posteriori error analysis (2017).

<a id="6">[6]</a> Bertrand, F. and Starke, G.: A posteriori error estimates by weakly symmetric stress reconstruction for the Biot problem (2021).

<a id="7">[7]</a> Ern, A. and Vohralik, M.: Polynomial-Degree-Robust A Posteriori Estimates in a Unified Setting for Conforming, Nonconforming, Discontinuous Galerkin, and Mixed Discretizations (2015).

<a id="8">[8]</a> Cai, Z. and Zhang, S.: Robust equilibrated residual error estimator for diffusion problems:
conforming elements (2012).

<a id="9">[9]</a> Bertrand et al.: Stabilization-free HHO a posteriori error control (2023).

<a id="10">[10]</a> Boffi et al.: Mixed finite element methods and applications (2013).

<a id="11">[11]</a> Kim, K.-Y.: Guaranteed A Posteriori Error Estimator for Mixed Finite Element Methods of Linear Elasticity with Weak Stress Symmetry (2011).

## License

dolfinx_eqlb is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

dolfinx_eqlb is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with dolfinx_eqlb. If not, see
<https://www.gnu.org/licenses/>.

