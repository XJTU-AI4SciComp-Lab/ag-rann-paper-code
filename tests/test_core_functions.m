function test_core_functions()
%TEST_CORE_FUNCTIONS Basic deterministic tests for shared numerical code.

root = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(root, "src"), "-begin");

x = (1:7)' / 8;
A = [ones(size(x)) x x .^ 2 x .^ 3 x .^ 4];
F = cos((1:7)');
coefficients = agrann.solve_output_layer(A, F, 0, 1);
assert(all(isfinite(coefficients)));
assert(size(coefficients, 1) == 5);

sample = [-2 -0.5 0 0.5 2];
for index = -1:3
    [activation, first_derivative, second_derivative] = agrann.wavelet_basis(index);
    assert(isequal(size(activation(sample)), size(sample)));
    assert(isequal(size(first_derivative(sample)), size(sample)));
    assert(isequal(size(second_derivative(sample)), size(sample)));
end

[interior_2d, boundary_2d] = agrann.uniform_points(2, 0, 1, 4, 3);
assert(isequal(size(interior_2d), [16 2]));
assert(isequal(size(boundary_2d), [12 2]));

[interior_3d, boundary_3d] = agrann.uniform_points(3, 0, 1, 3, 2);
assert(isequal(size(interior_3d), [27 3]));
assert(isequal(size(boundary_3d), [24 3]));

[points_1d, weights_1d] = agrann.gauss_legendre_1d(4, 0, 1);
assert(abs(sum(weights_1d) - 1) < 1e-14);
assert(size(points_1d, 1) == 4);

[points_2d, weights_2d] = agrann.gauss_legendre_2d(4, 0, 1, -1, 1);
assert(abs(sum(weights_2d) - 2) < 1e-14);
assert(isequal(size(points_2d), [16 2]));

fprintf("test_core_functions: PASS\n");
end
