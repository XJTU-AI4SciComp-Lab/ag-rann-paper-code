function [z, V] = AllenCahn(ind)

if nargin < 1
    ind = 1;
end    

num_inter = 300; 
num_bound = 1000;
num_neuron = 3000;
r_neuron = [5 70];
dimension = 2;
penalty_bound = 10;
num_neuron_add = 1500;
grad_random = 2;
max_it = [0, 15, 0, 5];
solu = @(x)agrann.problems.allen_cahn_initial_profile(x); 
E = 1e-4;
K = 5;

rng(ind)

left = 0;
right = 1;
bottom = -1;
top = 1;

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
eta0 = penalty_bound;

atv1 = @(t) exp(-t .^ 2 / 2);
diff1_atv1 = @(t) - t .* exp(-t .^ 2 / 2);
diff2_atv1 = @(t)(t .^ 2 .* exp(-t .^ 2 / 2) - exp(-t .^ 2 / 2));

[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis(0);
[atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(0);

W = [unifrnd(-r(1), r(1), 1, m); unifrnd(-r(2), r(2), 1, m)];
b = [unifrnd(left, right, 1, m); unifrnd(bottom, top, 1, m)];
b = - sum(b .* W, 1);

load(agrann.data_file('allen_cahn', 'AC_cosine.mat'))
[T, X] = meshgrid(tt, x);
err_pt = [T(:) X(:)];
U_err = uu(:);
V_err1 = atv1(err_pt * W + b);

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptt1 = linspace(left, right, (right - left) / (top - bottom) * nb)';
ptb1 = [zeros(nb, 1) linspace(bottom, top, nb)'];
ptb2 = [ptt1 bottom*ones(size(ptt1))];
ptb3 = [ptt1 top*ones(size(ptt1))];

Fi0 = zeros(size(pti(:, 1)));

Ab1 = atv1(ptb1 * W + b);
Fb1 = solu(ptb1);

Ab2 = atv1(ptb2 * W + b) - atv1(ptb3 * W + b);
Fb2 = zeros(size(Ab2, 1), 1);

Ab3 = W(2, :) .* diff1_atv1(ptb2 * W + b) - W(2, :) .* diff1_atv1(ptb3 * W + b);
Fb3 = zeros(size(Ab2, 1), 1);

Ur = zeros(m, 1);

V_atv1 = atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* diff1_atv1(pti * W + b);
V_diff2_atv1_2 = W(2, :) .^ 2 .* diff2_atv1(pti * W + b);

num = 1;
z = zeros(1, sum(max_it));

for i = 1 : max_it(1)
    Ai = V_diff1_atv1_1 - E * V_diff2_atv1_2 + K * (V_atv1 * Ur) .^ 2 .* V_atv1 - K * V_atv1;

    A = [Ai; eta0 * Ab1; Ab2; Ab3];
    F = [Fi0; eta0 * Fb1; Fb2; Fb3];

    Ur = agrann.solve_output_layer(A, F);

    err_point = V_err1 * Ur - U_err;
    err_L2_rel = sqrt(sum(err_point .^ 2) / sum(U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

    z(num) = err_L2_rel;
    num = num + 1;
end

for i = 1 : max_it(2)
    Ai = V_diff1_atv1_1 - E * V_diff2_atv1_2 + 3 * K * (V_atv1 * Ur) .^ 2 .* V_atv1 - K * V_atv1;

    Fi = 2 * K * (V_atv1 * Ur) .^ 3;

    A = [Ai; eta0 * Ab1; Ab2; Ab3];
    F = [Fi; eta0 * Fb1; Fb2; Fb3];

    Ur = agrann.solve_output_layer(A, F);

    err_point = V_err1 * Ur - U_err;
    err_L2_rel = sqrt(sum(err_point .^ 2) / sum(U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

    z(num) = err_L2_rel;
    num = num + 1;
end

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

ni = size(pti, 1);
[~, I] = sort(abs(A(1 : ni, :) * Ur - F(1 : ni)));
I = I(end - m2 + 1:end);
scatter(pti(I, 1), pti(I, 2), 20, 'red')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
axis([0, 1, -1, 1])

if grad_random == 0
    grad = [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (atv1(pti(I, :) * W + b) * Ur)';

Ab1(:, I2) = atv0(grad(1, :) .* (ptb1(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb1(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb1 * W + b) * Ur * G - V .* G);
Ab2(:, I2) = atv0(grad(1, :) .* (ptb2(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb2(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb2 * W + b) * Ur * G - V .* G) - ...
             atv0(grad(1, :) .* (ptb3(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb3(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb3 * W + b) * Ur * G - V .* G);
Ab3(:, I2) = atv0(grad(1, :) .* (ptb2(:, 1) - pti(I, 1)')) .* (grad(2, :) .* diff1_atv0(grad(2, :) .* (ptb2(:, 2) - pti(I, 2)'))) .* atv2(atv1(ptb2 * W + b) * Ur * G - V .* G) + ...
                atv0(grad(1, :) .* (ptb2(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb2(:, 2) - pti(I, 2)')) .* diff1_atv2(atv1(ptb2 * W + b) * Ur * G - V .* G) .* G .* (W(2, :) .* diff1_atv1(ptb2 * W + b) * Ur) - ...
             atv0(grad(1, :) .* (ptb3(:, 1) - pti(I, 1)')) .* (grad(2, :) .* diff1_atv0(grad(2, :) .* (ptb3(:, 2) - pti(I, 2)'))) .* atv2(atv1(ptb3 * W + b) * Ur * G - V .* G) - ...
                atv0(grad(1, :) .* (ptb3(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb3(:, 2) - pti(I, 2)')) .* diff1_atv2(atv1(ptb3 * W + b) * Ur * G - V .* G) .* G .* (W(2, :) .* diff1_atv1(ptb3 * W + b) * Ur);

Ur_new = [Ur; zeros(m - size(Ai, 2), 1)];

V_diff1_atv1_0 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1_0;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1_0;
V_diff2_atv1_0 = diff2_atv1(pti * W + b);
V_diff2_atv1_2 = W(2, :) .^ 2 .* V_diff2_atv1_0;

V_atv0_1 = atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_atv0_2 = atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));
V_diff1_atv0_1 = grad(1, :) .* diff1_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_diff1_atv0_2 = grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));
V_diff2_atv0_2 = grad(2, :) .^ 2 .* diff2_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));

V_atv2_0 = V_atv1 * Ur * G - V .* G;
V_atv2 = atv2(V_atv2_0);
V_diff1_atv2_0 = diff1_atv2(V_atv2_0) .* G;
V_diff1_atv2_1 = V_diff1_atv2_0 .* (V_diff1_atv1_1 * Ur);
V_diff1_atv2_2 = V_diff1_atv2_0 .* (V_diff1_atv1_2 * Ur);
V_diff2_atv2_0 = diff2_atv2(V_atv2_0) .* G .^ 2;
V_diff2_atv2_2 = V_diff2_atv2_0 .* (V_diff1_atv1_2 * Ur) .^ 2 + V_diff1_atv2_0 .* (V_diff2_atv1_2 * Ur);

V_err0_1 = atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)'));
V_err0_2 = atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)'));
V_err2 = atv2(V_err1 * Ur * G - V .* G);

for i = 1 : max_it(3)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Ai = zeros(size(pti, 1), m);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) - E * V_diff2_atv1_2(:, I1) + K * (Un .^ 2 - 1) .* V_atv1(:, I1);

    Ai(:, I2) = V_atv0_2 .* (V_diff1_atv0_1 .* V_atv2 + V_atv0_1 .* V_diff1_atv2_1) - ...
             E * (V_atv0_1 .* V_atv0_2 .* V_diff2_atv2_2 + 2 * V_atv0_1 .* V_diff1_atv0_2 .* V_diff1_atv2_2 + V_atv0_1 .* V_diff2_atv0_2 .* V_atv2) + ...
          K * (Un .^ 2 - 1) .* V_atv0_1 .* V_atv0_2 .* V_atv2;

    A = [Ai; eta0 * Ab1; Ab2; Ab3];
    F = [Fi0; eta0 * Fb1; Fb2; Fb3];

    Ur_new = agrann.solve_output_layer(A, F);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(sum(err_point .^ 2) / sum(U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

    z(num) = err_L2_rel;
    num = num + 1;
end

for i = 1 : max_it(4)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) - E * V_diff2_atv1_2(:, I1) + K * (3 * Un .^ 2 - 1) .* V_atv1(:, I1);

    Ai(:, I2) = V_atv0_2 .* (V_diff1_atv0_1 .* V_atv2 + V_atv0_1 .* V_diff1_atv2_1) - ...
             E * (V_atv0_1 .* V_atv0_2 .* V_diff2_atv2_2 + 2 * V_atv0_1 .* V_diff1_atv0_2 .* V_diff1_atv2_2 + V_atv0_1 .* V_diff2_atv0_2 .* V_atv2) + ...
          K * (3 * Un .^ 2 - 1) .* V_atv0_1 .* V_atv0_2 .* V_atv2;

    Fi = 2 * K * Un .^ 3;

    A = [Ai; eta0 * Ab1; Ab2; Ab3];
    F = [Fi; eta0 * Fb1; Fb2; Fb3];

    Ur_new = agrann.solve_output_layer(A, F);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(sum(err_point .^ 2) / sum(U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

    z(num) = err_L2_rel;
    num = num + 1;
end

V = atv1([T(:) X(:)] * W(:, I1) + b(:, I1)) * Ur_new(I1) + ...
    atv0(grad(1, :) .* (T(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (X(:) - pti(I, 2)')) .* atv2(atv1([T(:) X(:)]  * W + b) * Ur * G - V .* G) * Ur_new(I2);


toc

end
