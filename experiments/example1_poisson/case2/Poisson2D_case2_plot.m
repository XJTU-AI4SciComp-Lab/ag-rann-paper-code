function Poisson2D_case2_plot(local, grad_random)

if nargin < 1
    local = 1;
    grad_random = 10;
end    

num_inter = 300; 
num_bound = 3000;
num_neuron = 2000;
r_neuron = 15;
dimension = 2;
penalty_bound_init = 3000;
num_neuron_add = 1500;
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

% [atv1, diff1_atv1, diff2_atv1] = agrann.wavelet_basis(0);
[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis(0);
if local == 0
    [atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(-1);
else
    [atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(0);
end

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, left, right);

W = unifrnd(-r, r, d, m);
b = unifrnd(left, right, d, m);
b = - sum(b .* W, 1);

[pti, ptb] = agrann.uniform_points(2, left, right, ni, nb);
ni = size(pti, 1);

V_atv1 = atv1(pti * W + b);
V_diff2_atv1 = sum(W .^ 2, 1) .* diff2_atv1(pti * W + b);

U_err = solu(err_pt);
V_err1 = atv1(err_pt * W + b);

Ai = - V_diff2_atv1;
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F);

err_point = V_err1 * Ur - U_err;
err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

fprintf('L2_relative_error = %e, ', err_L2_rel);    
fprintf('Linf_relative_error = %e\n', err_Linf_rel);

X = reshape(err_pt(:, 1), 100, 100);
Y = reshape(err_pt(:, 2), 100, 100);

V = atv1([X(:) Y(:)] * W + b) * Ur;

figure('Units', 'centimeter', 'Position', [5 5 30 15]);
set(gcf, 'renderer', 'painters')

subplot(2, 3, 1)
agrann.visualization.draw_2d(X, Y, solu([X(:) Y(:)]));
title('exact solution')
subplot(2, 3, 2)
agrann.visualization.draw_2d(X, Y, V);
title('numerical solution')
subplot(2, 3, 3)
agrann.visualization.draw_2d(X, Y, abs(V - solu([X(:) Y(:)])));
title('absolute error')

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
% [~, I] = sort(abs(atv1(pti * W + b) * Ur - solu(pti)));
I = I(end - m2 + 1:end);
subplot(2, 3, 4)
scatter(pti(I, 1), pti(I, 2), 20, 'red')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 1 0 1])
title('error-indicator points')

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

V_err2 = atv2(V_err1 * Ur * G - V .* G);

Ai(:, I2) = ...
    - V_atv0_1 .* V_atv0_2 .* ... 
        (diff2_atv2(V_atv2_0) .* G .^ 2 .* ((V_diff1_atv1_1 * Ur) .^ 2 + (V_diff1_atv1_2 * Ur) .^ 2) + V_diff1_atv2_0 .* (V_diff2_atv1 * Ur)) - ...
      2 * V_diff1_atv2_0 .* (grad(1, :) .* diff1_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* V_atv0_2 .* (V_diff1_atv1_1 * Ur) + ...
                             V_atv0_1 .* grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')) .* (V_diff1_atv1_2 * Ur)) - ...
      atv2(V_atv2_0) .* (grad(1, :) .^ 2 .* diff2_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* V_atv0_2 + ...
                         V_atv0_1 .* grad(2, :) .^ 2 .* diff2_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')));

Ab(:, I2) = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb * W + b) * Ur * G - V .* G);

% agrann.visualization.plot_poisson_growth_basis(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2);

% Ai(I, :) = eta1 * Ai(I, :);
% Fi(I) = eta1 * Fi(I);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur_old = Ur;

% Ur(I2) = agrann.solve_output_layer(A(:, I2), F - A(:, I1) * Ur(I1));

Ur = agrann.solve_output_layer(A, F);

err_point = V_err1(:, I1) * Ur(I1) + atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)')) .* V_err2 * Ur(I2) - U_err;
err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

fprintf('L2_relative_error = %e, ', err_L2_rel);
fprintf('Linf_relative_error = %e\n', err_Linf_rel);

V = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)]  * W + b) * Ur_old * G - V .* G) * Ur(I2);

subplot(2, 3, 5)
agrann.visualization.draw_2d(X, Y, V)
title('numerical solution')
subplot(2, 3, 6)
agrann.visualization.draw_2d(X, Y, abs(V - solu([X(:) Y(:)])))
title('absolute error')

print(agrann.figure_file('ex1_case2.eps'), '-depsc2')

toc

end
