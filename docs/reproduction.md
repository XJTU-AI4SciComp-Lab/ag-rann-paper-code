# Reproduction Guide

Run all commands from the repository root after `startup`.

## Paper figures

| Paper figure | Experiment | Command |
|---|---|---|
| 1-4 | network/layer-growth schematics | source scripts under `experiments/paper_figures/` |
| 5 and 7 | Poisson Case 1 | `reproduce_figure_05_07` |
| 6 | frequency diagnostics within Poisson Case 1 | `run_experiment("example1_case1_growth")` |
| 8-9 | Poisson Case 2 | `reproduce_figure_08` |
| 10 | Poisson Case 3 | `reproduce_figure_10` |
| 11 | Poisson Case 4 | `reproduce_figure_11` |
| 12-14 | Poisson Case 5 | `reproduce_figure_12_14` |
| 15 | advection-reaction/domain splitting | `reproduce_figure_15` |
| 16-17 | Burgers Case 1 | `reproduce_figure_16_17` |
| 18-21 | Burgers Cases 2-4 | `reproduce_figure_18_21` |
| 22-23 | Allen-Cahn seed study and ensemble | `reproduce_figure_22_23` |

## Paper tables

| Paper table | Content | Main code |
|---|---|---|
| 1 | Example 1 parameters | experiment constants in `example1_poisson/` |
| 2 | Poisson Case 1 neuron growth | `Poisson1D_Growing` |
| 3 | Poisson Case 2 layer-growth variants | `reproduce_table_03` |
| 4 | FEM/adaptive FEM comparison | `reproduce_table_04_ifem` (external iFEM required) |
| 5 | Poisson Case 3 neuron growth | `Poisson2D_case3` |
| 6 | Poisson Case 4 layer growth | `reproduce_table_06` |
| 7 | Poisson Case 5 peak configurations | `Poisson2D` in `case5/` |
| 8 | Example 2 parameters | `AdvectionReaction*` constants |
| 9 | Example 3 parameters | case-specific Burgers constants |
| 10 | Burgers Case 1 neuron growth | `Burgers1D_Growing_plot` |
| 11 | Burgers Case 2 layer growth | `reproduce_table_11` |
| 12 | Burgers Case 4 neuron growth | `Burgers1D_case4_ALL` |
| 13 | Allen-Cahn parameters | `AllenCahn` constants |

## Runtime notes

- The quick-start example and tests finish in seconds.
- Full 2-D/3-D sweeps use large dense least-squares systems.
- Table 4 requires an external iFEM installation providing `squaremesh`,
  `femPoisson`, `afemPoisson`, and related functions.
- `run_all` includes all sweeps and the 100-seed Allen-Cahn ensemble and can
  take many hours with substantial memory use.
- The original scripts select CPU solves by default. Change local `GPU` flags
  only if a different numerical execution path is acceptable.

## Outputs

- Figures: `figures/generated/*.eps`
- Numerical results: `results/<experiment>/*.mat`
- Supplied references: `figures/reference/` and `results/reference/`
