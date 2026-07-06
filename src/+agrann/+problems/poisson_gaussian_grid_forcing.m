function z = poisson_gaussian_grid_forcing(x, n)
%POISSON_GAUSSIAN_GRID_FORCING Right-hand side for grid Gaussian peaks.

y = x(:, 2);
x = x(:, 1);
p1 = linspace(-1, 1, n + 1);
p1 = (p1(1:end-1)+p1(2:end)) / 2;
[p1, p2] = meshgrid(p1);
p1 = p1(:)'; p2 = p2(:)';
z = sum(-4000.*exp(- 1000.*(p1 - x).^2 - 1000.*(p2 - y).^2).*(1000.*p1.^2 - 2000.*p2.*y - 2000.*p1.*x + 1000.*p2.^2 + 1000.*x.^2 + 1000.*y.^2 - 1), 2);
end
