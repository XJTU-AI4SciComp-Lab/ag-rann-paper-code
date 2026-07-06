function z = poisson_spherical_shell_solution(x, S)
%POISSON_SPHERICAL_SHELL_SOLUTION Three-dimensional sharp Poisson solution.

z = x(:, 3);
y = x(:, 2);
x = x(:, 1);
z = 64 * (0.5 + atan(S * (0.25 ^ 2 - (x - 0.5) .^ 2 - (y - 0.5) .^ 2 - (z - 0.5) .^ 2)) / pi) ...
    .* x .* (1 - x) .* y .* (1 - y) .* z .* (1 - z);
end
