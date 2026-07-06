function [relative_l2_error, mean_solution] = average_ensemble()
%AVERAGE_ENSEMBLE Average the Allen-Cahn seed runs as in paper Figure 23.

ensemble = load(agrann.result_file("example4_allen_cahn", "result.mat"));
reference = load(agrann.data_file("allen_cahn", "AC_cosine.mat"));

mean_solution = sum(ensemble.V, 2) / size(ensemble.V, 2);
relative_l2_error = sqrt(sum((mean_solution - reference.uu(:)) .^ 2) ...
    / sum(reference.uu(:) .^ 2));

fprintf("Allen-Cahn ensemble relative_L2_error = %.16e\n", relative_l2_error);
end
