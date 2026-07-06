function z = advection_reaction_step_solution(x)
%ADVECTION_REACTION_STEP_SOLUTION Piecewise exact solution for example 2.

z = 1 - (x(:, 1) > 0.1) .* (x(:, 1) < 0.2) - (x(:, 1) > 0.3) .* (x(:, 1) < 0.4) - (x(:, 1) > 0.5) .* (x(:, 1) < 0.6) - (x(:, 1) > 0.7) .* (x(:, 1) < 0.8) - (x(:, 1) > 0.9) .* (x(:, 1) < 1);
end
