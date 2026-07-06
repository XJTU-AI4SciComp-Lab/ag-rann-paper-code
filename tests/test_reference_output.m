function test_reference_output()
%TEST_REFERENCE_OUTPUT Verify a deterministic reduced-size PDE experiment.

project_root = fileparts(fileparts(mfilename("fullpath")));
addpath(project_root, "-begin");
startup();

expected = [ ...
    9.561873613057290e-01, ...
    7.037208043864834e-01, ...
    0 ...
];
actual = run_experiment("example1_case2", 0, 0, 2, 10, 10);

assert(isequaln(actual, expected));
fprintf("test_reference_output: PASS\n");
end
