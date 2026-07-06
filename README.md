# Adaptive-Growth Randomized Neural Networks (AG-RaNN)

Official MATLAB implementation for the paper *Adaptive-Growth Randomized
Neural Networks: Algorithms, Theory, and PDE Applications*.

The repository provides reusable numerical components, runnable examples, and
scripts for reproducing the main figures and tables in the paper.

## Paper Information

- **Title:** Adaptive-Growth Randomized Neural Networks: Algorithms, Theory, and PDE Applications
- **Authors:** Haoning Dang, Fei Wang, Song Jiang
- **DOI / arXiv:** arXiv:2408.17225v3
- **Corresponding author:** Fei Wang, `feiwang.xjtu@xjtu.edu.cn`
- **Code maintainer:** Haoning Dang, `haoningdang.xjtu@stu.xjtu.edu.cn`

## Method Summary

AG-RaNN fixes the hidden-layer parameters and computes the output coefficients
by solving a linear least-squares problem. The method improves randomized
features without nonlinear training through four residual-driven mechanisms:

1. frequency-based parameter initialization;
2. neuron growth;
3. layer growth with localized basis functions;
4. domain splitting for discontinuous or piecewise-smooth solutions.

The numerical experiments cover Poisson, advection-reaction, Burgers, and
Allen-Cahn equations.

For computational efficiency, the growth procedures for different examples are
implemented as experiment-specific MATLAB drivers instead of a single generic
growth solver. The shared package contains reusable numerical building blocks,
while each experiment keeps its optimized assembly, residual evaluation, and
growth logic.

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- startup.m
|-- run_experiment.m
|-- src/+agrann/                 # shared MATLAB implementation
|-- examples/minimal_poisson_1d/ # quick deterministic example
|-- experiments/                 # paper experiments and reproduction scripts
|-- data/                        # required input data
|-- results/                     # generated and reference numerical results
|-- figures/                     # reference and generated figures
|-- tests/                       # MATLAB tests
`-- docs/                        # method and reproduction documentation
```

## Requirements

- MATLAB R2025b or a compatible release;
- Symbolic Math Toolbox;
- Statistics and Machine Learning Toolbox;
- Parallel Computing Toolbox and a CUDA-capable GPU are optional;
- the Table 4 finite-element comparison additionally requires iFEM.

The supplied experiments use CPU linear solves by default.

## Installation

Clone the repository, open MATLAB, and run:

```matlab
cd("path/to/ag-rann-paper-code")
startup
```

No additional setup is required for the core MATLAB experiments.

## Quick Start

```matlab
startup
run("examples/minimal_poisson_1d/run.m")
```

Verified output with MATLAB R2025b:

```text
minimal_poisson_1d relative_L2_error = 3.9489199889868488e-12
```

The result is saved to `results/minimal_poisson_1d/metrics.csv`.

## Reproduce Results

Run a single experiment through the project dispatcher:

```matlab
startup
run_experiment("example1_case2_figure")
run_experiment("example2_domain_split")
run_experiment("example3_case4_all")
```

Paper-numbered reproduction scripts are also available:

```matlab
reproduce_figure_05_07
reproduce_figure_15
reproduce_figure_18_21
reproduce_table_03
```

`run_all` reproduces all numerical figures, including the 100-seed Allen-Cahn
ensemble, and may take several hours. See
[docs/reproduction.md](docs/reproduction.md) for the complete mapping between
paper results and MATLAB entry points.

Generated EPS files are written to `figures/generated/`. Generated MAT files
are written to experiment-specific directories under `results/`.

## Tests

Run the self-contained MATLAB test suite:

```matlab
startup
addpath("tests")
run_tests
```

The suite tests the shared numerical routines, a deterministic PDE reference
run, and the minimal example. Validation details are available in
[docs/validation.md](docs/validation.md).

## Data

`data/allen_cahn/AC_cosine.mat` contains the Allen-Cahn reference solution.
The 100-seed ensemble output is generated locally and is not tracked because of
its size:

```matlab
run_experiment("example4_ensemble", 100)
```

See [data/README.md](data/README.md) for details. The iFEM comparison drivers
retain their upstream copyright notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use this code, please cite:

```bibtex
@article{dang2026agrann,
  title   = {Adaptive-Growth Randomized Neural Networks: Algorithms, Theory, and PDE Applications},
  author  = {Dang, Haoning and Wang, Fei and Jiang, Song},
  year    = {2026},
}
```

Machine-readable citation metadata is available in `CITATION.cff`.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
