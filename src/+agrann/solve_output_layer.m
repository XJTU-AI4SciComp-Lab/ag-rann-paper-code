function coefficients = solve_output_layer(A, F, use_gpu, use_qr)
%SOLVE_OUTPUT_LAYER Solve the linear output-layer least-squares system.
%
% This is the shared implementation of the former Solve_AF.m files.

if nargin < 3
    use_gpu = 0;
    use_qr = 1;
end

if use_gpu == 1 && gpuDeviceCount > 0
    coefficients = gpuArray(A) \ gpuArray(F);
else
    if use_qr == 1
        opts.RECT = true;
        coefficients = linsolve(A, F, opts);
    else
        coefficients = A \ F;
    end
end
end
