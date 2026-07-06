function reproduce_figure_22_23()
%REPRODUCE_FIGURE_22_23 Reproduce the Allen-Cahn ensemble results.

run_experiment("example4_ensemble", 100);
run_experiment("example4_average");
end
