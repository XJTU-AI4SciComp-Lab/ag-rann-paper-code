function z = burgers_fan_solution(x)
%BURGERS_FAN_SOLUTION Piecewise fan profile for Burgers' equation.

z = (x(:, 2) > 0) .* (x(:, 2) < x(:, 1)) .* x(:, 2) ./ (x(:, 1) + 1e-16) + (x(:, 2) >= x(:, 1));
end
