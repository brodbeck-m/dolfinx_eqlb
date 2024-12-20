# <a name="dolfinxeqlb"></a> dolfinx for flux equilibration (dolfinx_eqlb)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--4498-d45815.svg)](https://doi.org/10.18419/darus-4498)

Author: Maximilian Brodbeck

This library contains an add-on to FEniCSx enabling local flux equilibration strategies. The resulting H(div) conforming fluxes can be used for the construction of adaptive finite element solvers for the Poisson problem [[5]](#5)[[8]](#8), elasticity [[1]](#1)[[9]](#9) or poro-elasticity [[2]](#2)[[10]](#10).  

The equilibration process relies on so called patches, groups of all cells, connected with one node of the mesh. On each patch a constrained minimisation problem is solved [[8]](#8). In order to improve computational efficiency, a so called semi-explicit strategy [[3]](#3)[[6]](#6) is also implemented. The solution procedure is thereby split into two steps: An explicit determination of an H(div) function, fulfilling the minimisation constraints, followed by an unconstrained minimisation on a reduced, patch-wise ansatz space. If equilibration is applied to elasticity - the stress tensor has a distinct symmetry - an additional constrained minimisation step after the row wise reconstruction of the tensor [[1]](#1) is implemented.

* [Features](#features)
* [Installation](#installation)
    * [Docker](#installation_docker)
    * [Source](#installation_source)
    * [Getting started](#installation_getting-started)
* [Documentation](#documentation)
    * [Local solvers](#doc_local-solver)
    * [The equilibrator](#doc_equilibrator)
    * [Equilibrated fluxes for a-posteriori error estimation](#doc_error-estimation)
* [How to cite](#how-to-cite)
* [Literature](#literature)
* [License](#license)

# <a id="features"></a> Features
dolfinx_eqlb supports flux equilibration on two-dimensional domains with arbitrary triangular grids. It further includes the following features
- A local projector into arbitrary function-spaces with cell-wise support
- A hierarchic Raviart-Thomas element based on Boffi, Brezzi and Fortin [[4]](#4)
- Boundary conditions for H(div) spaces on general boundaries
- Flux equilibration based on Ern and Vohralik (FluxEqlbEV) or a semi-explicit strategy (FluxEqlbSE)
- Stress equilibration considering distinct symmetry properties in a weak sense

# <a id="installation"></a> Installation

## <a id="installation_docker"></a> Docker
A docker image of dolfinx_eqlb can be created based on the docker image of DOLFINx. Therefore the following steps are required:
1. Clone this repository using the command:

```shell
git clone git@github.com:brodbeck-m/dolfinx_eqlb.git
```

2. Download the basic Docker image of DOLFINx:

```shell
docker pull dolfinx/dolfinx:v0.6.0-r1
```

3. Build a Docker image containing dolfinx_eqlb

```shell
cd docker
./build_image.sh 
```

4. Start the docker container
```shell
# Use the provided start-script
./docker/launch-container.sh

# Use docker directly
docker run -ti --rm -v "$(pwd)":/root/shared -w /root/shared brodbeck-m/dolfinx_eqlb:release
```

Alternatively, a ready-to-use image of the latest release can be downloaded from [DaRUS](https://doi.org/10.18419/darus-4498). Based on the .tar.gz the container can be created
```shell
docker load --input dockerimage-dolfinx_eqlb-v1.2.0.tar.gz
```

## <a id="installation_source"></a> Source
To install the latest version (main branch), you need to install release 0.6.0 of DOLFINx. Easiest way to install DOLFINx is to use docker. The required DOLFINx docker images goes under the name dolfinx/dolfinx:v0.6.0-r1.

To install the dolfinx_eqlb-library run the following code from this directory:
```shell
# Correct include of the eigen-library in pybind11
expot PYBIND_EIGEN_PATH=/usr/local/lib/python3.10/dist-packages/pybind11/include/pybind11/eigen.h
sed -i 's%<Eigen/Core>%<eigen3/Eigen/Core>%' ${PYBIND_EIGEN_PATH} && \
sed -i 's%<Eigen/SparseCore>%<eigen3/Eigen/SparseCore>%' ${PYBIND_EIGEN_PATH}

# Install dolfinx_eqlb
cmake -G Ninja -B build-dir -DCMAKE_BUILD_TYPE=Release cpp/
ninja -C build-dir install
pip3 install python/ -v --upgrade
```

## <a id="installation_getting-started"></a> Getting started
In order to check the correctness of the installation two basic demos - the equilibration of an [Poisson-type flux](https://github.com/brodbeck-m/dolfinx_eqlb/blob/main/python/demo/poisson/demo_reconstruction.py) as well as a [weakly symmetric stress tensor from linear elasticity](https://github.com/brodbeck-m/dolfinx_eqlb/blob/main/python/demo/elasticity/demo_reconstruction.py) - should be tested. No errors should be reported!
```shell
# Equilibration for a Poisson problem
cd ./root/dolfinx_eqlb/python/demo/poisson
python3 demo_reconstruction.py  

# Equilibration for linear elasticity
cd ./root/dolfinx_eqlb/python/demo/elasticity
python3 demo_reconstruction.py  
```

Further information on the python-interface of dolfinx_eqlb are provided in the [documentation](#documentation) or demonstrated in [further examples](https://github.com/brodbeck-m/dolfinx_eqlb/tree/main/python/demo).

# <a id="documentation"></a> Documentation
Flux equilibration can either be used to improve the accuracy of dual quantity - e.g. the flux for a Poisson or heat equation or the stress in elasticity - by a post-processing step or as a basis for a-posteriori error estimation. Incorporating equilibration into a solution procedure required the following four steps:
1. Solve the primal problem based on Lagrangian finite elements of degree $k$.
2. Calculate the approximated dual quantity from the primal field.
3. Project the right-hand-side (RHS) of the primal problem as well as the approximated dual quantity into discontinuous Lagrange spaces of order $m \geq k$. The projection can be done locally.
4. Equilibrate the dual quantity in an Raviart-Thomas space of order $m$.

The algorithmic structure of the equilibration itself is described in [AddSource]. A short description of the relevant Python interfaces of the library is given below. Complete examples for the Poisson equation and linear elasticity can be found in the [demo section](https://github.com/brodbeck-m/dolfinx_eqlb/tree/main/python/demo). 

## <a id="doc_local-solver"></a> Local projections
Projecting an arbitrary function $\mathrm{f}$ into a discontinuous finite element space $\mathrm{V}$ requires the solution of
```math
\left(\mathrm{u},\;\mathrm{v}\right) = (\mathrm{f},\mathrm{v})
```
for all $\mathrm{v}\in\mathrm{V}$. As the function space $\mathrm{V}$ is discontinuous, the solution on each finite element can be computed independently. Assuming ```f_ufl``` to be the ufl-representation of a function, the following code snippet shows the local projection:
```python
from dolfinx.fem import FunctionSpace
from dolfinx_eqlb.lsolver import local_projection

# Initialise target function space
V_proj = FunctionSpace(domain, ("DG", m - 1))

# Project f_ufl into fe-function
f_proj = local_projection(V_proj, [f_ufl])
```
```f_proj``` will be a list of functions, with the same length as the second argument of 'local_projection'. This allows the simultaneous projection of multiple function (as long as they have the same target function space), which is beneficial from a performance perspective, as the system matrix has to be factorised only once. Due to the symmetric and positive definite system matrix, a Cholesky decomposition is used.

## <a id="doc_equilibrator"></a> The equilibrator
Based on projections of the approximated dual quantity and the RHS the equilibrator itself can be initialised. In order to improve efficiency, multiple RHS can be equilibrated at the same time. Assuming that for each $\boldsymbol{\varsigma}^\mathrm{R}_h$ a divergence condition of the form
```math
\nabla\cdot\boldsymbol{\varsigma}^\mathrm{R}_h = \Pi_{m-1}\mathrm{f}
```
holds. $\Pi_{m-1}\left(\bullet\right)$ denotes the projection into a discontinous Lagrange space of order $m-1$. 

Having lists of DOLFINx functions with the projected RHS ```list_rhs```$`=\left\{\mathrm{f}_i\right\}`$
and the projected dual quantities ```list_sigmah```$`=\left\{\boldsymbol{\varsigma}\left(\mathrm{u}_h\right)\big\vert_i\right\}`$ the equilibrator is initialised as follows:
```python
from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

# Initialise equilibrated (approach by Ern and Vohralik [7])
equilibrator = FluxEqlbEV (m, domain , list_rhs , list_sigmah)

# Initialise equilibrated (semi-explicit approach [8,9])
equilibrator = FluxEqlbSE (m, domain , list_rhs , list_sigmah)
```
The semi-explicit equilibrator can be initialised with two optional arguments ```equilibrate_stress``` and ```estimate_korn_constant```. When the first one is set, the first ```gdim``` fluxes are treated as rows of a stress tensor and symmetry is enforced weakly [[1]](#1). The one enables the evaluation of the cells Korn constants (only for 2D) based on [[7]](#7). The Korn constants, stored within a $\mathrm{DP}_0$ function, can be extracted by the appropriate getter method:
```python
equilibrator.get_korn_constants()
```

Before the actual equilibration can be performed, boundary data namely the boundary facets on $\Gamma_\mathrm{D}$ and $\Gamma_\mathrm{N}$ as well as the normal traces of the flux on $\Gamma_\mathrm{N}$ , are required. While the facets lists are of type ```NDArray```, the (different) normal traces are stored in a list.

Assuming a domain with Dirichlet BCs on facets with ```facet_tags```$`\in\left\{1,2\right\}`$ and Neumann BCs for ```facet_tags```$`\in\left\{3,4\right\}`$, the following code snippet shows how to specify them:
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
The function ```fluxbc``` has thereby the optional arguments ```requires_projection``` and ```quadrature_degree```. They are required when the traction lies not in the polynomial space of order $m-1$. Setting ```requires_projection``` enforces an $\mathrm{L}^2$-projection when the boundary DOFs are evaluated. Specifying ```quadrature_degree```, prescribes the quadrature degree for the evaluation of the linear form of the projection. 

With these information provided for each simultaneously equilibrated flux the equilibration can be solved:
```python
# Set boundary conditions
equilibrator.set_boundary_conditions(list_dfcts, list_bcs)

# Solve equilibration
equilibrator.equilibrate_fluxes()
```

## <a id="doc_error-estimation"></a> Equilibrated fluxes for a-posteriori error estimation
Based on equilibrated fluxes reliable error estimates for different problem classes can be constructed. Showcases for the Poisson problem (estimate by Ern and Vohralik [[8]](#8)) and linear elasticity (following Bertrand et al. [[1]](#1)) are provided in the [demo section](https://github.com/brodbeck-m/dolfinx_eqlb/tree/main/python/demo). For both problems the equilibration- and error estimation process is demonstrated on a unit-square with manufactured solution on a series of uniformly refined meshes:
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

# <a id="how-to-cite"></a> How to cite
dolfinx_eqlb is a research software. The latest release can be cited via [DaRUS](https://doi.org/10.18419/darus-4498), or - if citations of individual files or code lines are required - via Software Heritage <a href="https://archive.softwareheritage.org/swh:1:rel:c14e312283cb4768028660317eadbadbaef448e7;origin=https://github.com/brodbeck-m/dolfinx_eqlb;visit=swh:1:snp:6c4a8ff43ab40b7d4a0589a6c0fa3afcc60d7c3e">
    <img src="https://archive.softwareheritage.org/badge/swh:1:rel:c14e312283cb4768028660317eadbadbaef448e7/" alt="Archived | swh:1:rel:c14e312283cb4768028660317eadbadbaef448e7"/>
</a>.

If you are using using dolfinx_eqlb please also cite the related publication

**Adaptive finite element methods based on flux equilibration using FEniCSx**. preprint (2024). arXiv: [2410.09764](https://arxiv.org/abs/2410.09764)
```bib
@misc{DolfinxEqlb2024,
    title = {Adaptive finite element methods based on flux and stress equilibration using FEniCSx},
    author={Maximilian Brodbeck, Fleurianne Bertrand and Tim Ricken},
    year = {2024},
    eprint={2410.09764},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2410.09764}
}
```

# <a id="literature"></a> Literature
<a id="1">[1]</a> Bertrand, F., Kober, B., Moldenhauer, M. and Starke, G.: Weakly symmetric stress equilibration and a posteriori error estimation for linear elasticity. Comput. Math. Appl. (2021) doi: [10.1002/num.22741](https://doi.org/10.1002/num.22741)

<a id="2">[2]</a> Bertrand, F. and Starke, G.: A posteriori error estimates by weakly symmetric stress reconstruction for the Biot problem. Numer. Methods Partial Differ. Equ. (2021) doi: [10.1016/j.camwa.2020.10.011](https://doi.org/10.1016/j.camwa.2020.10.011)

<a id="3">[3]</a> Bertrand, F., Carstensen, C., Gräßle, B. and Tran, N.T.: Stabilization-free HHO a posteriori error control. Numer. Math. (2023) doi: [10.1007/s00211-023-01366-8](https://doi.org/10.1007/s00211-023-01366-8)

<a id="4">[4]</a> Boffi, D., Brezzi, F. and Fortin, M.: Mixed finite element methods and applications. Springer Heidelberg, Berlin (2013). 

<a id="5">[5]</a> Braess, D. and Schöberl, J.: Equilibrated Residual Error Estimator for Edge Elements. Math. Comput. 77, 651–672 (2008)

<a id="6">[6]</a> Cai, Z. and Zhang, S.: Robust equilibrated residual error estimator for diffusion problems: conforming elements. SIAM J. Numer. Anal. (2012). doi: [10.1137/100803857](https://doi.org/10.1137/100803857)

<a id="7">[7]</a> Kim, K.-Y.: Guaranteed A Posteriori Error Estimator for Mixed Finite Element Methods of Linear Elasticity with Weak Stress Symmetry. SIAM J. Numer. Anal. (2015) doi: [10.1137/110823031](https://doi.org/10.1137/110823031)

<a id="8">[8]</a> Ern, A and Vohralı́k, M.: Polynomial-Degree-Robust A Posteriori Estimates in a Unified Setting for Conforming, Nonconforming, Discontinuous Galerkin, and Mixed Discretizations. SIAM J. Numer. Anal. (2015) doi: [10.1137/130950100](https://doi.org/10.1137/130950100)

<a id="9">[9]</a> Prager, W. and Synge, J.L.: Approximations in elasticity based on the concept of function space. Q. J. Mech. Appl. Math. 5, 241–269 (1947)

<a id="10">[10]</a> Riedlbeck, R., Di Pietro, D.A., Ern, A., Granet, S. and Kazymyrenko, K.: Stress and flux reconstruction in Biot’s poro-elasticity problem with application to a posteriori error analysis. Comput. Math. Appl. (2017) doi: [10.1016/j.camwa.2017.02.005](https://doi.org/10.1016/j.camwa.2017.02.005)

# <a id="license"></a> License

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

