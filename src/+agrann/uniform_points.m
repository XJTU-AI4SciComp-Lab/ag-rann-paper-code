function [interior_points, boundary_points] = uniform_points(dimension, left, right, num_interior, num_boundary)
%UNIFORM_POINTS Construct the original tensor-product collocation grids.

epsilon = 1e-10;

switch dimension
    case 2
        points = linspace(left + epsilon, right - epsilon, num_interior);
        [point_x, point_y] = meshgrid(points, points);
        interior_points = [point_x(:) point_y(:)];
        points = linspace(left, right, num_boundary)';
        boundary_points = [[points left * ones(num_boundary, 1)]; ...
            [right * ones(num_boundary, 1) points]; ...
            [points right * ones(num_boundary, 1)]; ...
            [left * ones(num_boundary, 1) points]];
    case 3
        points = linspace(left + epsilon, right - epsilon, num_interior);
        [point_x, point_y, point_z] = meshgrid(points, points, points);
        interior_points = [point_x(:) point_y(:) point_z(:)];
        points = linspace(left, right, num_boundary);
        [point_1, point_2] = meshgrid(points, points);
        point_1 = point_1(:);
        point_2 = point_2(:);
        boundary_points = [[point_1 point_2 left * ones(num_boundary * num_boundary, 1)]; ...
            [point_1 point_2 right * ones(num_boundary * num_boundary, 1)]; ...
            [point_1 left * ones(num_boundary * num_boundary, 1) point_2]; ...
            [point_1 right * ones(num_boundary * num_boundary, 1) point_2]; ...
            [left * ones(num_boundary * num_boundary, 1) point_1 point_2]; ...
            [right * ones(num_boundary * num_boundary, 1) point_1 point_2]];
    otherwise
        error("agrann:uniform_points:UnsupportedDimension", ...
            "Only dimensions 2 and 3 are supported.");
end
end
