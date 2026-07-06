function z = poisson_gaussian_ring_forcing(x, n)
%POISSON_GAUSSIAN_RING_FORCING Right-hand side for ring Gaussian peaks.

y = x(:, 2);
x = x(:, 1);
c = linspace(pi/4, pi*9/4, n+1);
c(end) = [];
p1 = 0.5 * sqrt(2) * cos(c);
p2 = 0.5 * sqrt(2) * sin(c);
z = sum(-4000.*exp(- 1000.*(p1 - x).^2 - 1000.*(p2 - y).^2).*(1000.*p1.^2 - 2000.*p2.*y - 2000.*p1.*x + 1000.*p2.^2 + 1000.*x.^2 + 1000.*y.^2 - 1), 2);
end
