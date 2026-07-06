function [points, weights] = gauss_legendre_3d(N, left, right, bottom, top)
%GAUSS_LEGENDRE_3D Build the original cubic tensor-product quadrature.

if nargin < 1
    N = 30;
    left = 0;
    right = 1;
    bottom = 0;
    top = 1;
end

syms x

points = sort(double(root(legendreP(N, x), x)));
weights = 2 * (1 - points .^ 2) ./ (N + 1) ^ 2 ./ legendreP(N + 1, points) .^ 2;

[point_x, point_y, point_z] = meshgrid( ...
    (right - left) / 2 * points + (right + left) / 2, ...
    (top - bottom) / 2 * points + (top - bottom) / 2, ...
    (top - bottom) / 2 * points + (top - bottom) / 2);
points = [point_x(:) point_y(:) point_z(:)];
[weight_x, weight_y, weight_z] = meshgrid(weights);
weights = weight_x .* weight_y .* weight_z;
weights = weights(:) / 8 * (right - left) * (top - bottom) * (top - bottom);
end
