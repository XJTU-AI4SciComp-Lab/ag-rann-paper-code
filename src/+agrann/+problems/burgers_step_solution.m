function z = burgers_step_solution(x)
%BURGERS_STEP_SOLUTION Step profile for Burgers' equation.

z = (sign(x(:, 1) / 2 - x(:, 2)) + 1) / 2;
end
