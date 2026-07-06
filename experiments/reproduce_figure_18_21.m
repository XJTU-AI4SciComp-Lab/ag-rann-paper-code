function reproduce_figure_18_21()
%REPRODUCE_FIGURE_18_21 Reproduce Burgers Cases 2-4.

run_experiment("example3_case2");
run_experiment("example3_case3");
run_experiment("example3_case4_known");
run_experiment("example3_case4_all");
end
