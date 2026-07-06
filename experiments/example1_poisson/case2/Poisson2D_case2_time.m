function z = Poisson2D_case2_time(local, fixed, grad_random, num_neuron, num_neuron_add)

if nargin < 1
    local = 0;
    fixed = 0;
    grad_random = 10;
    num_neuron = 1500;
    num_neuron_add = 500;
end    

num_inter = 300; 
num_bound = 3000;
r_neuron = 15;
dimension = 2;
penalty_bound_init = 3000;
S = 120;
scale1 = 1;
scale2 = 1;

rng(1)

solu = @(x)agrann.problems.poisson_sharp_solution(x, S);
rhs = @(x)agrann.problems.poisson_sharp_forcing(x, S);
left = 0;
right = 1;

close all
format long e
warning off

ni = num_inter; 
nb = num_bound;
m = num_neuron;
m_add = num_neuron_add;
r = r_neuron;
d = dimension;
eta0 = penalty_bound_init;

tic

atv1 = @(x)tanh(x);
diff1_atv1 = @(x) 1 - tanh(x) .^ 2;
diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis(0);
if local == 0
    [atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(-1);
else
    [atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(0);
end

W = unifrnd(-r, r, d, m);
b = unifrnd(left, right, d, m);
b = - sum(b .* W, 1);

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

Ur = agrann.solve_output_layer(A, F);

I2 = find(Ur == 0);
m2 = size(I2, 1);
fprintf('m2 = %d\n', m2);

if m2 < m_add
    I2 = [I2; (m + 1 : m + m_add - m2)'];
    m = m + m_add - m2;
    m2 = m_add;
end

I1 = (1 : m)';
I1(ismember(I1, I2)) = [];

[~, I] = sort(abs(Ai(1 : ni, :) * Ur - Fi(1 : ni)));
I = I(end - m2 + 1:end);

if grad_random == 0
    grad = scale1 * [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (V_atv1(I, :) * Ur)';

grad = scale2 * grad;

V_diff1_atv1_0 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1_0;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1_0;

V_atv0_1 = atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_atv0_2 = atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));

V_atv2_0 = V_atv1 * Ur * G - V .* G;
V_diff1_atv2_0 = diff1_atv2(V_atv2_0) .* G;

Ai(:, I2) = ...
    - V_atv0_1 .* V_atv0_2 .* ... 
        (diff2_atv2(V_atv2_0) .* G .^ 2 .* ((V_diff1_atv1_1 * Ur) .^ 2 + (V_diff1_atv1_2 * Ur) .^ 2) + V_diff1_atv2_0 .* (V_diff2_atv1 * Ur)) - ...
      2 * V_diff1_atv2_0 .* (grad(1, :) .* diff1_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* V_atv0_2 .* (V_diff1_atv1_1 * Ur) + ...
                             V_atv0_1 .* grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')) .* (V_diff1_atv1_2 * Ur)) - ...
      atv2(V_atv2_0) .* (grad(1, :) .^ 2 .* diff2_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* V_atv0_2 + ...
                         V_atv0_1 .* grad(2, :) .^ 2 .* diff2_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')));

Ab(:, I2) = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb * W + b) * Ur * G - V .* G);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

if fixed == 1
    Ur(I2) = agrann.solve_output_layer(A(:, I2), F - A(:, I1) * Ur(I1));
else
    Ur = agrann.solve_output_layer(A, F);
end

z = toc;

end