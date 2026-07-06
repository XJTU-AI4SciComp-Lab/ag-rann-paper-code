function z = poisson_sharp_solution(x, sharpness)
%POISSON_SHARP_SOLUTION Exact 2-D sharp Poisson solution.

y = x(:, 2);
x = x(:, 1);
z = 16 * (0.5 + atan(sharpness * (0.25 ^ 2 - (x - 0.5) .^ 2 ...
    - (y - 0.5) .^ 2)) / pi) .* x .* (x - 1) .* y .* (y - 1);
end
