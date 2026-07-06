function [activation, first_derivative, second_derivative] = wavelet_basis(index)
%WAVELET_BASIS Return the basis function and its first two derivatives.
%
% This is the shared implementation used by all 1-D/2-D experiments.

if nargin < 1
    index = 1;
end

switch index
    case -1
        activation = @(t) ones(size(t));
        first_derivative = @(t) zeros(size(t));
        second_derivative = @(t) zeros(size(t));
    case 0
        activation = @(t) exp(-t .^ 2 / 2);
        first_derivative = @(t) -t .* exp(-t .^ 2 / 2);
        second_derivative = @(t) t .^ 2 .* exp(-t .^ 2 / 2) - exp(-t .^ 2 / 2);
    case 1
        activation = @(t) t / sqrt(2 * pi) .* exp(-t .^ 2 / 2);
        first_derivative = @(t) (exp(-t .^ 2 / 2) - t .^ 2 .* exp(-t .^ 2 / 2)) / sqrt(2 * pi);
        second_derivative = @(t) (t .^ 3 .* exp(-t .^ 2 / 2) - 3 * t .* exp(-t .^ 2 / 2)) / sqrt(2 * pi);
    case 2
        activation = @(t) cos(1.75 * t) .* exp(-t .^ 2 / 2);
        first_derivative = @(t) -(7 * sin((7 * t) / 4) .* exp(-t .^ 2 / 2)) / 4 ...
            - t .* cos((7 * t) / 4) .* exp(-t .^ 2 / 2);
        second_derivative = @(t) t .^ 2 .* cos((7 * t) / 4) .* exp(-t .^ 2 / 2) ...
            - (65 * cos((7 * t) / 4) .* exp(-t .^ 2 / 2)) / 16 ...
            + (7 * t .* sin((7 * t) / 4) .* exp(-t .^ 2 / 2)) / 2;
    case 3
        activation = @(t) 2 / sqrt(3) * pi ^ (-0.25) * (1 - t .^ 2) .* exp(-t .^ 2 / 2);
        first_derivative = @(t) t .* exp(-t .^ 2 / 2) .* (t .^ 2 - 1) ...
            - 2 * t .* exp(-t .^ 2 / 2);
        second_derivative = @(t) 2 / sqrt(3) * pi ^ (-0.25) ...
            * (exp(-t .^ 2 / 2) .* (t .^ 2 - 1) - 2 * exp(-t .^ 2 / 2) ...
            + 4 * t .^ 2 .* exp(-t .^ 2 / 2) ...
            - t .^ 2 .* exp(-t .^ 2 / 2) .* (t .^ 2 - 1));
    otherwise
        error("agrann:wavelet_basis:UnknownIndex", ...
            "Unsupported wavelet-basis index: %g.", index);
end
end
