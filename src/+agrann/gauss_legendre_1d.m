function [points, weights] = gauss_legendre_1d(N, left, right)
%GAUSS_LEGENDRE_1D Build the original N-point Gauss-Legendre rule.

if nargin < 1
    N = 200;
    left = 0;
    right = 1;
end

syms x

points = sort(double(root(legendreP(N, x), x)));
weights = 2 * (1 - points .^ 2) ./ (N + 1) ^ 2 ./ legendreP(N + 1, points) .^ 2;
points = (right - left) / 2 * points + (right + left) / 2;
weights = weights / 2 * (right - left);
end
