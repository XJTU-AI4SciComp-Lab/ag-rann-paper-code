function [z, V] = generate_ensemble(num_seeds)
%GENERATE_ENSEMBLE Reproduce the 100-seed Allen-Cahn ensemble.

if nargin < 1
    num_seeds = 100;
end

z = zeros(20, num_seeds);
V = zeros(512 * 201, num_seeds);
for seed = 1:num_seeds
    [z(:, seed), V(:, seed)] = AllenCahn(seed);
end

save(agrann.result_file("example4_allen_cahn", "result.mat"), "z", "V");
end
