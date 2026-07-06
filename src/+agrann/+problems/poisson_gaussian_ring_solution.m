function z = poisson_gaussian_ring_solution(x, n)
%POISSON_GAUSSIAN_RING_SOLUTION Gaussian peaks arranged on a ring.

y = x(:, 2);
x = x(:, 1);
c = linspace(pi/4, pi*9/4, n+1);
c(end) = [];
p1 = 0.5 * sqrt(2) * cos(c);
p2 = 0.5 * sqrt(2) * sin(c);
z = sum(exp(-1000 * ((x - p1) .^ 2 + (y - p2) .^ 2)), 2);
end
