# Validation

The repository is validated with MATLAB R2025b Update 5.

## Automated tests

Run:

```matlab
startup
addpath("tests")
run_tests
```

The suite checks:

1. output-layer least-squares solves;
2. wavelet bases and their derivatives;
3. 2-D and 3-D collocation-point generation;
4. 1-D and 2-D Gauss-Legendre quadrature;
5. a deterministic reduced-size Poisson experiment;
6. the minimal one-dimensional Poisson example.

The deterministic PDE reference output is:

```text
[9.561873613057290e-01, 7.037208043864834e-01, 0]
```

The reference check uses exact `isequaln` comparison in the verified MATLAB
environment.

## Source checks

All MATLAB files are scanned with `checkcode` for parsing and syntax errors.
The main paper experiments are also inspected with
`matlab.codetools.requiredFilesAndProducts` to verify dependency resolution.

## Reference artifacts

The repository includes small MAT reference results and the available EPS
figures. Their SHA-256 hashes are listed in
[`reference_artifacts.csv`](reference_artifacts.csv).

The large Allen-Cahn ensemble output is generated with:

```matlab
run_experiment("example4_ensemble", 100)
```
