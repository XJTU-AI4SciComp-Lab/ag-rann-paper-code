# Experiments

The subdirectories correspond directly to the four numerical examples in the
paper. Use `run_experiment` from the repository root so only the selected
experiment directory is added to the MATLAB path. Problem definitions are
implemented in `src/+agrann/+problems`; experiment folders keep only
experiment-specific drivers and visualization helpers.

For efficiency, the growth procedure is implemented separately for each
example rather than through a single generic growth program. The shared `src`
package supplies common numerical components, and each experiment keeps its own
optimized assembly and growth logic.

The `reproduce_figure_*.m` files provide paper-numbered entry points.
`run_all.m` executes every numerical entry point and is intentionally expensive.
