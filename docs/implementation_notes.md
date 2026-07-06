# Implementation Notes

## Shared MATLAB package

Reusable routines are implemented in the `agrann` MATLAB package:

| Component | MATLAB interface | Purpose |
|---|---|---|
| output-layer solver | `agrann.solve_output_layer` | CPU/GPU linear least-squares solve |
| wavelet bases | `agrann.wavelet_basis` | basis values and first/second derivatives |
| normalized wavelet basis | `agrann.wavelet_basis_normalized` | normalized Gaussian variant used in 3-D |
| collocation grids | `agrann.uniform_points` | 2-D and 3-D interior/boundary points |
| 1-D quadrature | `agrann.gauss_legendre_1d` | Gauss-Legendre points and weights |
| 2-D quadrature | `agrann.gauss_legendre_2d` | tensor-product quadrature |
| 3-D quadrature | `agrann.gauss_legendre_3d` | cubic tensor-product quadrature |
| benchmark definitions | `agrann.problems.*` | shared exact solutions and forcing terms |
| visualization | `agrann.visualization.*` | shared surface and basis-function plots |

Problem-specific equations are implemented in `agrann.problems.*`; experiment
parameters, matrix assembly, growth loops, and plotting choices remain under
`experiments/`.

## Experiment-specific growth implementations

The growth algorithms are not wrapped in one generic driver. Each numerical
example has its own MATLAB implementation of the growth procedure. This layout
is intentional: the examples use different PDE operators, residual terms,
boundary treatments, basis localization rules, and plotting outputs. Keeping the
growth code close to each experiment avoids extra abstraction overhead and
preserves the efficient matrix assembly used for the reported results.

The shared package therefore provides reusable building blocks, while the
experiment drivers implement the example-specific growth logic.

## Experiment path isolation

Several experiments intentionally use short local visualization helper names
such as `Draw2D`. Adding every experiment directory at once would make function
resolution depend on MATLAB path order.

`run_experiment` therefore adds only the selected experiment directory and
removes it after execution. Users should call `startup` and use
`run_experiment` instead of adding the entire repository with `genpath`.

## Reproducibility

The implementation preserves the numerical choices used for the reported
results:

- fixed random seeds;
- deterministic random-number call order;
- symbolic construction of Gauss-Legendre points;
- `linsolve(..., opts.RECT=true)` for the default output solve;
- experiment-specific activation scaling;
- the normalized wavelet variant used by the 3-D Poisson experiment;
- the plotting commands and EPS output drivers used by the paper scripts.

## Output locations

- generated MAT files: `results/<experiment>/`;
- generated EPS files: `figures/generated/`;
- Allen-Cahn reference input: `data/allen_cahn/AC_cosine.mat`;
- small reference results: `results/reference/`;
- reference figures: `figures/reference/`.
