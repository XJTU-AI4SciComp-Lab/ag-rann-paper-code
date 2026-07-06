function test_problem_definitions()
%TEST_PROBLEM_DEFINITIONS Deterministic checks for shared problem formulas.

root = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(root, "src"), "-begin");

points_1d = [0; 0.25; 0.5; 0.75; 1];
points_2d = [0.1 0.2; 0.4 -0.3; 0.9 0.7];
points_3d = [0.2 0.3 0.4; 0.6 0.5 0.7];

assert_close(agrann.problems.poisson_multiscale_1d_solution(points_1d, 3), ...
    [0; 3.333333333333333e-01; -2.041077998578922e-16; -3.333333333333335e-01; -5.715018396020981e-16]);
assert_close(agrann.problems.poisson_multiscale_1d_forcing(points_1d, 3), ...
    [0; 1.315947253478577e+01; -1.144215120191699e-13; -1.315947253478595e+01; -2.352893064056170e-13]);

assert_close(agrann.problems.poisson_gaussian_ring_solution(points_2d, 4), ...
    [2.669190215540973e-109; 1.928749847963877e-22; 1.383896526736895e-87]);
assert_close(agrann.problems.poisson_gaussian_ring_forcing(points_2d, 4), ...
    [-2.658513454678810e-103; -3.780349702009200e-17; -1.101581635282568e-81]);

assert_close(agrann.problems.poisson_gaussian_grid_solution(points_2d, 2), ...
    [2.669190215541276e-109; 1.928749847963918e-22; 1.383896526736738e-87]);
assert_close(agrann.problems.poisson_gaussian_grid_forcing(points_2d, 2), ...
    [-2.658513454679035e-103; -3.780349702009281e-17; -1.101581635282443e-81]);

assert_close(agrann.problems.poisson_spherical_shell_solution(points_3d, 10), ...
    [1.497375579579668e-01; 4.351200732430802e-01]);
assert_close(agrann.problems.poisson_spherical_shell_forcing(points_3d, 10), ...
    [-1.386662802492830e+00; 2.296569630038222e+01]);

assert_close(agrann.problems.advection_reaction_step_solution([0.15 0; 0.25 0; 0.35 0; 0.95 0]), ...
    [0; 1; 0; 0]);
assert_close(agrann.problems.advection_reaction_step_forcing([0.15 0; 0.25 0; 0.35 0; 0.95 0]), ...
    [0; 1; 0; 0]);

assert_close(agrann.problems.burgers_logistic_solution(points_2d, 0.1), ...
    [3.208213008246070e-01; 9.241418199787566e-01; 2.227001388253090e-01]);
assert_close(agrann.problems.burgers_fan_solution([0.5 0.25; 0.5 0.75; -0.5 0.1]), ...
    [4.999999999999999e-01; 1.000000000000000e+00; 1.000000000000000e+00]);
assert_close(agrann.problems.burgers_step_solution([0.5 0.1; 0.5 0.3]), ...
    [1; 0]);

assert_close(agrann.problems.allen_cahn_initial_profile(points_2d), ...
    [3.236067977499790e-02; 5.290067270632258e-02; -2.880147736233117e-01]);

fprintf("test_problem_definitions: PASS\n");
end

function assert_close(actual, expected)
actual = actual(:);
expected = expected(:);
assert(isequal(size(actual), size(expected)));
scale = max(1, norm(expected, inf));
assert(norm(actual - expected, inf) <= 1e-12 * scale);
end
