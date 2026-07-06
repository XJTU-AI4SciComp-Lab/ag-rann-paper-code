function z = burgers_logistic_solution(x, E)
%BURGERS_LOGISTIC_SOLUTION Exact viscous Burgers profile.

x(:, 1) = - x(:, 1) / 2;
z = 1 ./ (1 + exp(sum(x, 2) / E / 2));
end
