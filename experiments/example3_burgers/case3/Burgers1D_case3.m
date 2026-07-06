function Burgers1D_case3 

if nargin < 1
    num_inter = 200; 
    num_bound = 200;
    num_neuron = 1000;
    r_neuron = [3 15];
    dimension = 2;
    penalty_bound_init = 5;
    num_neuron_add = 1000;
    lambda = 0;
    scale1 = 1;
    scale2 = 1;
    grad_random = 10;
    max_it = [10, 0, 0, 5];
    solu = @(x)agrann.problems.burgers_fan_solution(x); % case 1
end    

rng(1)

rhs = @(x)agrann.problems.zero_forcing(x);

GPU = 0;
QR = 1;

left = 0;
right = 1;
bottom = -1;
top = 2;

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

it_err = zeros(1, sum(max_it));
num_it_err = 1;

atv1 = @(x)tanh(x);
diff1_atv1 = @(x) 1 - tanh(x) .^ 2;

[atv0, diff1_atv0, ~] = agrann.wavelet_basis(0);
[atv2, diff1_atv2, ~] = agrann.wavelet_basis(0);

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, bottom, top);

W = [unifrnd(-r(1), r(1), 1, m); unifrnd(-r(2), r(2), 1, m)];
b = [unifrnd(left, right, 1, m); unifrnd(bottom, top, 1, m)];
b = - sum(b .* W, 1);

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptb = [zeros(nb, 1) linspace(bottom, top, nb)'];

U_err = solu(err_pt);
V_err1 = atv1(err_pt * W + b);

Fi0 = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

Ur = zeros(m, 1);

V_atv1 = atv1(pti * W + b);
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;

for i = 1 : max_it(1)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2;

    A = [Ai; eta0 * Ab];
    F = [Fi0; eta0 * Fb];

    if lambda ~= 0
        F = A.' * F;
        A = A.' * A + lambda * eye(size(A, 2));
    end

    Ur = agrann.solve_output_layer(A, F, GPU, QR);

    err_point = V_err1 * Ur - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

for i = 1 : max_it(2)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur);

    Fi = Fi0 + (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];
    
    if lambda ~= 0
        F = A.' * F;
        A = A.' * A + lambda * eye(size(A, 2));
    end

    Ur = agrann.solve_output_layer(A, F, GPU, QR);
    
    err_point = V_err1 * Ur - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

[X, Y] = meshgrid(linspace(left, right, 500), linspace(bottom, top, 500));

gcf = figure('Units', 'centimeter', 'Position', [5 5 30 15]);
set(gcf, 'renderer', 'painters')

subplot(2, 3, 1)
Draw2D(X, Y, solu([X(:) Y(:)]));
V = atv1([X(:) Y(:)] * W + b) * Ur;
title('exact solution')

subplot(2, 3, 2)
Draw2D(X, Y, V);
title('numerical solution')
subplot(2, 3, 3)
Draw2D(X, Y, abs(V - solu([X(:) Y(:)])));
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

[~, I] = sort((W(1, :) .* diff1_atv1(pti * W + b) * Ur) .^ 2 + (W(2, :) .* diff1_atv1(pti * W + b) * Ur) .^ 2);
I = I(end - m2 + 1:end);
subplot(2, 3, 4)
scatter(pti(I, 1), pti(I, 2), 20, 'red')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
axis([0, 1, -1, 2])
title('error-indicator points')


if grad_random == 0
    grad = scale1 * [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (atv1(pti(I, :) * W + b) * Ur)';

grad = scale2 * grad;

Ab(:, I2) = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb * W + b) * Ur * G - V .* G);

Fi = rhs(pti);

Ur_new = [Ur; zeros(m - size(Ai, 2), 1)];

V_diff1_atv1_0 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1_0;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1_0;

V_atv0_1 = atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_atv0_2 = atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));
V_diff1_atv0_1 = grad(1, :) .* diff1_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_diff1_atv0_2 = grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));

V_atv2_0 = V_atv1 * Ur * G - V .* G;
V_atv2 = atv2(V_atv2_0);
V_diff1_atv2_0 = diff1_atv2(V_atv2_0) .* G;
V_diff1_atv2_1 = V_diff1_atv2_0 .* (V_diff1_atv1_1 * Ur);
V_diff1_atv2_2 = V_diff1_atv2_0 .* (V_diff1_atv1_2 * Ur);

V_err0_1 = atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)'));
V_err0_2 = atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)'));
V_err2 = atv2(V_err1 * Ur * G - V .* G);

for i = 1 : max_it(3)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Ai = zeros(size(pti, 1), m);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1);

    Ai(:, I2) = V_atv0_2 .* (V_diff1_atv0_1 .* V_atv2 + V_atv0_1 .* V_diff1_atv2_1) + ...
          Un .* V_atv0_1 .* (V_diff1_atv0_2 .* V_atv2 + V_atv0_2 .* V_diff1_atv2_2);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F, GPU, QR);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

for i = 1 : max_it(4)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Unx = V_diff1_atv1_2(:, I1) * Ur_new(I1) + V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 * Ur_new(I2) + ...
                                               V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2 * Ur_new(I2);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1) + V_atv1(:, I1) .* Unx;

    Ai(:, I2) = V_diff1_atv0_1 .* V_atv0_2 .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_1 + ...
          Un .* (V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2) + ...
          V_atv0_1 .* V_atv0_2 .* V_atv2 .* Unx;

    Fi = Fi0 + Un .* Unx;

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F, GPU, QR);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

V = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur_new(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)]  * W + b) * Ur * G - V .* G) * Ur_new(I2);

subplot(2, 3, 5)
Draw2D(X, Y, V);
title('numerical solution')
subplot(2, 3, 6)
Draw2D(X, Y, abs(V - solu([X(:) Y(:)])));
title('absolute error')

print(agrann.figure_file('ex3_case3.eps'), '-depsc2')

toc

end
