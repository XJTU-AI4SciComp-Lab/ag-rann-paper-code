function z = poisson_gaussian_grid_solution(x, n)
%POISSON_GAUSSIAN_GRID_SOLUTION Gaussian peaks arranged on a tensor grid.

y = x(:, 2);
x = x(:, 1);
p1 = linspace(-1, 1, n + 1);
p1 = (p1(1:end-1)+p1(2:end)) / 2;
[p1, p2] = meshgrid(p1);
p1 = p1(:)'; p2 = p2(:)';
z = sum(exp(-1000 * ((x - p1) .^ 2 + (y - p2) .^ 2)), 2);
end
