function run_tests()
%RUN_TESTS Run the fast test suite.

test_directory = fileparts(mfilename("fullpath"));
project_root = fileparts(test_directory);
addpath(project_root, "-begin");
addpath(test_directory, "-begin");
startup();

test_core_functions();
test_problem_definitions();
test_reference_output();
run(fullfile(project_root, "examples", "minimal_poisson_1d", "run.m"));

fprintf("All fast tests passed.\n");
end
