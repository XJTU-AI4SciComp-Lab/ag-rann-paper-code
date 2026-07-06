function z = allen_cahn_initial_profile(x)
%ALLEN_CAHN_INITIAL_PROFILE Initial profile used by the Allen-Cahn example.

x = x(:, 2);
z = x .^ 2 .* cos(pi * x);
end
