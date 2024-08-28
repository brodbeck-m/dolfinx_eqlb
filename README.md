# dolfinx for flux equilibration (dolfinx_eqlb)
## Description
dolfinx_eqlb is an open source library, extending the FEniCSx Project finite element solver (https://fenicsproject.org) by local flux equilibration strategies. The resulting H(div) conforming fluxes can be used for the construction of adaptive finite element solvers for the Poisson problem [[1]](#1), elasticity [[2]](#2)[[3]](#3)[[4]](#4) or poro-elasticity [[5]](#5)[[6]](#6).  

The flux equilibration relies on so called patches, groups of all cells, connected with one node of the mesh. On each patch a constrained minimisation problem is solved [[7]](#7). In order to improve computational efficiency, additionally a so called semi-explicit strategy [[8]](#8)[[9]](#9) is implemented. The solution procedure is thereby split into two steps: An explicit determination of an H(div) function, fulfilling the minimisation constraints, followed by an unconstrained minimisation on a reduced, patch-wise ansatz space. If equilibration is applied to elasticity -- stress tensors have distinct symmetry properties -- an additional constrained minimisation step, after the row wise reconstruction of the tensor [[3]](#3)[[4]](#4) is implemented.

## Features
dolfinx_eqlb supports flux equilibration on two-dimensional domains with arbitrary triangular grids. It further includes the following features
- A local projector into arbitrary function-spaces with cell-wise support
- A hierarchic Raviart-Thomas element based on Boffi, Brezzi and Fortin [[10]](#10)
- Boundary conditions for H(div) spaces on general boundaries
- Flux equilibration based on Ern and Vohralik (FluxEqlbEV) or a semi-explicit strategy (FluxEqlbSE)
- Stress equilibration considering distinct symmetry properties in a weak sense

## Getting started
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

## Equilibrated fluxes for a-posteriori error estimation
Based on equilibrated fluxes reliable error estimates for different problem classes can be constructed. Showcases for the Poisson problem (estimated by Ern and Vohralik [[2]](#2)) and linear elasticity (idea from Bertrand et al. [[3]](#3) transferred to a displacement-based formulation of linear elasticity) are provided in the demo section:

```shell
./launch-container.sh

# Equilibration for a Poisson problem
cd ./root/dolfinx_eqlb/python/demo/poisson
python3 demo_error_estimation.py  

# Equilibration for linear elasticity
cd ./root/dolfinx_eqlb/python/demo/elasticity
python3 demo_error_estimation.py  
```

Therein, a manufactured solution is solved on a rectangular domain, discretised by a series of uniformly refined meshes. The actual error, contributions of the error estimate and convergence ratios are reported in a csv-file. 

This approach can now be transferred to other problems.

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

