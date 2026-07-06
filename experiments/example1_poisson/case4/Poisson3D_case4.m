function z = Poisson3D_case4(num_neuron, num_neuron_add)

if nargin < 1
    num_neuron = 2000;
    num_neuron_add = 2000;
end    

num_inter = 80; 
num_bound = 40;
r_neuron = 5;
dimension = 3;
penalty_bound_init = 5000;
S = 120;
scale1 = 1;
scale2 = 1;
grad_random = 8;

rng(1)

solu = @(x)agrann.problems.poisson_spherical_shell_solution(x, S);
rhs = @(x)agrann.problems.poisson_spherical_shell_forcing(x, S);

GPU = 0;
QR = 1;

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

[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis_normalized(0);
[atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis_normalized(0);

[err_pt, err_wt] = agrann.gauss_legendre_3d(40, left, right, left, right);

W = unifrnd(-r, r, d, m);
b = unifrnd(left, right, d, m);
b = - sum(b .* W, 1);

[pti, ptb] = agrann.uniform_points(3, left, right, ni, nb);
ni = size(pti, 1);

U_err = solu(err_pt);
V_err1 = atv1(err_pt * W + b);

Ai = - sum(W .^ 2, 1) .* diff2_atv1(pti * W + b);
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F, GPU, QR);

err_point = V_err1 * Ur - U_err;
err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

fprintf('L2_relative_error = %e, ', err_L2_rel);    
fprintf('Linf_relative_error = %e\n', err_Linf_rel);

z(1) = err_L2_rel;

X = reshape(err_pt(:, 1), 40, 40, 40);
Y = reshape(err_pt(:, 2), 40, 40, 40);
Z = reshape(err_pt(:, 3), 40, 40, 40);

gcf = figure('Units', 'centimeter', 'Position', [5 5 30 15]);
set(gcf, 'renderer', 'painters')

subplot(2, 3, 1)
Draw3D(X, Y, Z, solu([X(:) Y(:) Z(:)]));
title('exact solution')

V = atv1([X(:) Y(:) Z(:)] * W + b) * Ur;

subplot(2, 3, 2)
Draw3D(X, Y, Z, V);
title('numerical solution')
subplot(2, 3, 3)
Draw3D(X, Y, Z, abs(V - solu([X(:) Y(:) Z(:)])));
title('absolute error')

I2 = find(Ur == 0);
m2 = size(I2, 1);
fprintf('m2 = %d\n', m2);

z(3) = m2;

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
scatter3(pti(I, 1), pti(I, 2), pti(I, 3), 20, 'red')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 1 0 1 0 1])
title('error-indicator points')

if grad_random == 0
    grad = scale1 * [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(3, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (atv1(pti(I, :) * W + b) * Ur)';

grad = scale2 * grad;

V_atv0_1 = atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_atv0_2 = atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));
V_atv0_3 = atv0(grad(3, :) .* (pti(:, 3) - pti(I, 3)'));
V_atv2_0 = atv1(pti * W + b) * Ur * G - V .* G;
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv2_0 = diff1_atv2(V_atv2_0);
V_diff2_atv2_0 = diff2_atv2(V_atv2_0);

V_err2 = atv2(V_err1 * Ur * G - V .* G);

Ai2 = - V_atv0_1 .* V_atv0_2 .* V_atv0_3 .* ...
          (V_diff2_atv2_0 .* (G .* (V_diff1_atv1 .* W(1, :) * Ur)) .^ 2 + ...
           V_diff2_atv2_0 .* (G .* (V_diff1_atv1 .* W(2, :) * Ur)) .^ 2 + ...
           V_diff2_atv2_0 .* (G .* (V_diff1_atv1 .* W(3, :) * Ur)) .^ 2 + ...
           V_diff1_atv2_0 .* G .* (- Ai * Ur)) - ...
      2 * grad(1, :) .* diff1_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* V_atv0_2 .* V_atv0_3 .* V_diff1_atv2_0 .* G .* (V_diff1_atv1 .* W(1, :) * Ur) - ...
      2 * grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')) .* V_atv0_1 .* V_atv0_3 .* V_diff1_atv2_0 .* G .* (V_diff1_atv1 .* W(2, :) * Ur) - ...
      2 * grad(3, :) .* diff1_atv0(grad(3, :) .* (pti(:, 3) - pti(I, 3)')) .* V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_0 .* G .* (V_diff1_atv1 .* W(3, :) * Ur) - ...
      grad(1, :) .^ 2 .* diff2_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* V_atv0_2 .* V_atv0_3 .* atv2(V_atv2_0) - ...
      grad(2, :) .^ 2 .* diff2_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')) .* V_atv0_1 .* V_atv0_3 .* atv2(V_atv2_0) - ...
      grad(3, :) .^ 2 .* diff2_atv0(grad(3, :) .* (pti(:, 3) - pti(I, 3)')) .* V_atv0_1 .* V_atv0_2 .* atv2(V_atv2_0);

Ab2 = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv0(grad(3, :) .* (ptb(:, 3) - pti(I, 3)')) .* atv2(atv1(ptb * W + b) * Ur * G - V .* G);

Ai(:, I2) = Ai2;
Ab(:, I2) = Ab2;

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur_old = Ur;

Ur = agrann.solve_output_layer(A, F, GPU, QR);

save(agrann.result_file('example1_poisson_case4', 'UUU.mat'), 'Ur', 'Ur_old')

err_point = V_err1(:, I1) * Ur(I1) + atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)')) .* atv0(grad(3, :) .* (err_pt(:, 3) - pti(I, 3)')) .* V_err2 * Ur(I2) - U_err;
err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

fprintf('L2_relative_error = %e, ', err_L2_rel);
fprintf('Linf_relative_error = %e\n', err_Linf_rel);

z(2) = err_L2_rel;

V = atv1([X(:) Y(:) Z(:)] * W(:, I1) + b(:, I1)) * Ur(I1) + atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv0(grad(3, :) .* (Z(:) - pti(I, 3)')) .* atv2(atv1([X(:) Y(:) Z(:)]  * W + b) * Ur_old * G - V .* G) * Ur(I2);

subplot(2, 3, 5)
Draw3D(X, Y, Z, V) 
title('numerical solution')
subplot(2, 3, 6)
Draw3D(X, Y, Z, abs(V - solu([X(:) Y(:) Z(:)])))
title('absolute error')

print(agrann.figure_file('ex1_case4.eps'), '-depsc2')

toc

end
