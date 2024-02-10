# dolfinx for flux equilibration (dolfinx_eqlb)
## Description
DOLFINx_eqlb is an open source library, extending the FEniCSx Project finite element solver (https://fenicsproject.org) by (patch-local) flux equilibration strategies. The resulting H(div) conforming fluxes, which strongly full-fill the Neumann boundary conditions of the primal problem, can be used for the construction of space-time adaptive finite element solvers. 

The basic idea of these reconstructions can be traced back to Prager and Synge [[1]](#1) and can be applied to the poisson problem [[2]](#2), incompressible elasticity [[3]](#3)[[4]](#4) or poro-elasticity [[5]](#5)[[6]](#6). The reconstruction itself can be performed either by a patch-wise constrained minimisation problem (Ern and Vohralik [[7]](#7)) or based on a semi-explicit procedure, where only a unconstrained minimisation on a patch-wise, divergence free function spaces is required [[8]](#8)[[9]](#9). For stress tensors distinct symmetry properties have to be considered. This can be done in a weak sene, which requires an additional constrained minimisation step, after the row wise reconstruction of the tensor [[3]](#3)[[4]](#4).

## Features
DOLFINx_eqlb supports flux equilibration on two-dimensional domain with arbitrary triangular grids. It further includes twi following features
- A local projector into arbitrary function-spaces
- A hierarchic Raviart-Thomas element based on Boffi, Brezzi and Fortin [[10]](#10)
- Boundary conditions for H(div) spaces on general boundaries
- Flux equilibration based on Ern and Vohralik (FluxEqlbEV) or a semi-explicit strategy (FluxEqlbSE)
- Stress equilibration considering distinct symmetry properties in a weak sense

## Getting started
1. Clone this repository using the command:

```shell
git clone https://github.tik.uni-stuttgart.de/brodbeck/dolfinx_eqlb
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
4. Try out a demo
```shell
./launch-container.sh
cd ./Programs/dolfinx_eqlb/python/demo
python3 demo_reconstruction_poisson.py  
```

## Literature
<a id="1">[1]</a> Prager, W. and Synge, J. L.: Approximations in elasticity based on the concept of function space (1947).

<a id="2">[2]</a> Braess, D. and Schöberl, J.: Equilibrated Residual Error Estimator for Edge Elements (2008).

<a id="3">[3]</a> Bertrand et al.: Weakly symmetric stress equilibration and a posteriori error estimation for linear elasticity (2021).

<a id="4">[4]</a> Bertrand et al.: Weakly symmetric stress equilibration for hyperelastic material models (2020).

<a id="5">[5]</a> Riedlbeck et al.: Stress and flux reconstruction in Biot’s poro-elasticity problem with application to a posteriori error analysis (2017).

<a id="6">[6]</a> Bertrand, F. and Starke, G.: A posteriori error estimates by weakly symmetric stress reconstruction for the Biot problem (2021).

<a id="7">[7]</a> Ern, A. and Vohralik, M.: Polynomial-Degree-Robust A Posteriori Estimates in a Unified Setting for Conforming, Nonconforming, Discontinuous Galerkin, and Mixed Discretizations (2015).

<a id="8">[8]</a> Cai, Z., Zhang, S.: Robust equilibrated residual error estimator for diffusion problems:
conforming elements (2012).

<a id="9">[9]</a> Bertrand et al.: Stabilization-free HHO a posteriori error control (2023).

<a id="10">[10]</a> Boffi et al.: Mixed finite element methods and applications (2013).

