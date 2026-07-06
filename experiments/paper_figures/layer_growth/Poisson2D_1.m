function Poisson2D_1

if nargin < 1
    num_inter = 300; 
    num_bound = 3000;
    num_neuron = 1000;
    r_neuron = 15;
    dimension = 2;
    penalty_bound_init = 3000;
    num_neuron_add = 1000;
    lambda = 0;
    S = 120;
    scale1 = 1;
    scale2 = 1;
    grad_random = 0;
end    

rng(1)

solu = @(x)agrann.problems.poisson_sharp_solution(x, S);
rhs = @(x)agrann.problems.poisson_sharp_forcing(x, S);
left = 0;
right = 1;

GPU = 0;
QR = 1;

close all
format long e
warning off

tic

ni = num_inter; 
nb = num_bound;
m = num_neuron;
m_add = num_neuron_add;
r = r_neuron;
d = dimension;
eta0 = penalty_bound_init;

atv1 = @(x)tanh(x);
diff1_atv1 = @(x) 1 - tanh(x) .^ 2;
diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis(0);
[atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(0);

W = unifrnd(-r, r, d, m);
b = unifrnd(left, right, d, m);
b = - sum(b .* W, 1);
W(:, 1) = 0;
b(:, 1) = 0;

[pti, ptb] = agrann.uniform_points(2, left, right, ni, nb);
ni = size(pti, 1);

V_atv1 = atv1(pti * W + b);
V_diff2_atv1 = sum(W .^ 2, 1) .* diff2_atv1(pti * W + b);

Ai = - V_diff2_atv1;
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

if lambda ~= 0
    F = A.' * F;
    A = A.' * A + lambda * eye(size(A, 2));
end

Ur = agrann.solve_output_layer(A, F, GPU, QR);

t = linspace(left, right, 200);
[X, Y] = meshgrid(t, t);

I2 = find(Ur == 0);
m2 = size(I2, 1);
fprintf('m2 = %d\n', m2);

if m2 < m_add
    m2 = m_add;
end

[~, I] = sort(abs(Ai(1 : ni, :) * Ur - Fi(1 : ni)));
I = I(end - m2 + 1:end);

if grad_random == 0
    grad = [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = scale1 * sqrt(sum(grad .^ 2, 1));
V = (V_atv1(I, :) * Ur)';

grad = scale2 * grad;

% GrowingBasis(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2);
% GrowingBasis1(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2);
GrowingBasis2(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2);

toc

end