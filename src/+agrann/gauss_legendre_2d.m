function [points, weights] = gauss_legendre_2d(N, left, right, bottom, top)
%GAUSS_LEGENDRE_2D Build the tensor-product Gauss-Legendre rule.

if nargin < 1
    N = 100;
    left = 0;
    right = 1;
    bottom = 0;
    top = 1;
end

syms x

points = sort(double(root(legendreP(N, x), x)));
weights = 2 * (1 - points .^ 2) ./ (N + 1) ^ 2 ./ legendreP(N + 1, points) .^ 2;

[point_x, point_y] = meshgrid( ...
    (right - left) / 2 * points + (right + left) / 2, ...
    (top - bottom) / 2 * points + (top + bottom) / 2);
points = [point_x(:) point_y(:)];
weights = weights * weights';
weights = weights(:) / 4 * (right - left) * (top - bottom);
end
