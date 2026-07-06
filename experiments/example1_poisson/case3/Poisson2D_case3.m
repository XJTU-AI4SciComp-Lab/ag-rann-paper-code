function Poisson2D_case3

if nargin < 1
    num_inter = 300; 
    num_bound = 3000;
    num_neuron = 200 * [3 2 1 1 1 1 1];
    num_neuron_add = 1500;
    r_neuron_range = 100;
    penalty_bound_init = 3000;
    S = 120;
    pre_weight = 100;
    scale1 = 1;
    scale2 = 1;
    grad_random = 10;
end    

rng(1);

Fs = 100;

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
r = r_neuron_range;
eta0 = penalty_bound_init;
pw = pre_weight;
res = zeros(3, length(m));

% atv1 = @(x)tanh(x);
% diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis(0);
[atv1, diff1_atv1, diff2_atv1] = agrann.wavelet_basis(0);
[atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(0);

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, left, right);

[pti, ptb] = agrann.uniform_points(2, left, right, ni, nb);
ni = size(pti, 1);

T = 1 / Fs;
[tx, ty] = meshgrid(left : T : right);
L = length(left : T : right); 

P_rhs = abs(fft2(reshape(rhs([tx(:) ty(:)]), L, L)));
P_rhs = P_rhs(1 : (L + 1) / 2, 1 : (L + 1) / 2);

Wf = linspace(0, r, pw + 1);
Wf = Wf(2 : end);
[Wf1, Wf2] = meshgrid(Wf);
Wf = [Wf1(:)'; Wf2(:)'];
D2X = - sum(Wf .^ 2, 1) .* diff2_atv1([tx(:) - (left + right) / 2, ty(:) - (left + right) / 2] * Wf);
P_weight = abs(fft2(reshape(D2X, L, L, pw ^ 2)));
P_weight = P_weight(1 : ((L + 1) / 2), 1 : ((L + 1) / 2), :);

[~, maxrhs] = max(P_rhs, [], "all"); 
[~, maxPwt] = max(P_weight, [], [1, 2], 'linear'); 
maxPwt = reshape(maxPwt, 1, pw ^ 2) - (0 : pw ^ 2 - 1) * ((L + 1) / 2) ^ 2;
maxPwt = [maxPwt - floor(maxPwt / ((L + 1) / 2)) * ((L + 1) / 2); floor(maxPwt / ((L + 1) / 2)) + 1];
maxrhs = [maxrhs - floor(maxrhs / ((L + 1) / 2)) * ((L + 1) / 2); floor(maxrhs / ((L + 1) / 2)) + 1];
I = find(maxPwt(1, :) > maxrhs(1) & maxPwt(2, :) > maxrhs(2));
[~, r] = min(sum(maxPwt(:, I) .^ 2, 1));
r = Wf(:, I(r));
fprintf('r = (%.6f, %.6f)\n', r(1), r(2))

% % % figure
% % % for i = 1 : pw
% % %     nexttile
% % %     hold on
% % %     plot(fs(1:d), P_weight(1:d, i), LineWidth=2) 
% % %     scatter(fs(maxPwt(i)), P_weight(maxPwt(i), i), 'filled')
% % %     title(['r = ', num2str(Wf(i))])
% % %     % xlabel("f (Hz)")
% % %     % ylabel("|P(f)|")
% % % end

W = [unifrnd(-r(1), r(1), 1, m(1)); unifrnd(-r(2), r(2), 1, m(1))];
b = unifrnd(left, right, 2, m(1));
b = - sum(b .* W, 1);

U_err = solu(err_pt);

Ai = - sum(W .^ 2, 1) .* diff2_atv1(pti * W + b);
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F);

err_L2_rel = sqrt(err_wt' * (atv1(err_pt * W + b) * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);    
res(1:2, 1) = r;
res(3, 1) = err_L2_rel;

X_lhs = reshape(- sum(W .^ 2, 1) .* diff2_atv1([tx(:) ty(:)] * W + b) * Ur, L, L);
Y_lhs = fft2(X_lhs);
P_lhs = abs(Y_lhs);
P_lhs = P_lhs(1 : (L + 1) / 2, 1 : (L + 1) / 2);

for i = 2 : length(m)
    [~, maxrhs] = max(abs(P_lhs - P_rhs), [], "all"); 
    maxrhs = [maxrhs - floor(maxrhs / ((L + 1) / 2)) * ((L + 1) / 2); floor(maxrhs / ((L + 1) / 2)) + 1];
    I = [find(maxPwt(1, :) > maxrhs(1) & maxPwt(2, :) > maxrhs(2)) pw ^ 2];
    [~, r] = min(sum(maxPwt(:, I) .^ 2, 1));
    r = Wf(:, I(r));
    fprintf('r = (%.6f, %.6f)\n', r(1), r(2))

    W_add = [unifrnd(-r(1), r(1), 1, m(i)); unifrnd(-r(2), r(2), 1, m(i))];
    b_add = unifrnd(left, right, 2, m(i));
    b_add = - sum(b_add .* W_add, 1);
    W = [W, W_add];
    b = [b, b_add];

    Ai = [Ai, - sum(W_add .^ 2, 1) .* diff2_atv1(pti * W_add + b_add)];
    
    Ab = [Ab, atv1(ptb * W_add + b_add)];
    
    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];
    
    Ur = agrann.solve_output_layer(A, F);
    
    err_L2_rel = sqrt(err_wt' * (atv1(err_pt * W + b) * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
    fprintf('L2_relative_error = %e\n', err_L2_rel);  
    res(1:2, i) = r;
    res(3, i) = err_L2_rel;

    X_lhs = reshape(- sum(W .^ 2, 1) .* diff2_atv1([tx(:) ty(:)] * W + b) * Ur, L, L);
    Y_lhs = fft2(X_lhs);
    P_lhs = abs(Y_lhs);
    P_lhs = P_lhs(1 : (L + 1) / 2, 1 : (L + 1) / 2);
    
end

for i = 1 : length(m)
    fprintf('(%.2f,%.2f) & ', res(1, i), res(2, i))
end
fprintf('\n');
for i = 1 : length(m)
    fprintf('%.4e & ', res(3, i))
end
fprintf('\n');

t = linspace(left, right, 200);
[X, Y] = meshgrid(t, t);

% agrann.visualization.draw_2d(X, Y, solu([X(:) Y(:)]));

V = atv1([X(:) Y(:)] * W + b) * Ur;

figure('Units', 'centimeter', 'Position', [5 5 50 8]);
set(gcf, 'renderer', 'painters')

subplot(1, 4, 1)
agrann.visualization.draw_2d(X, Y, V);
subplot(1, 4, 2)
agrann.visualization.draw_2d(X, Y, abs(V - solu([X(:) Y(:)])));

I2 = find(Ur == 0);
m2 = size(I2, 1);
fprintf('m2 = %d\n', m2);

m = sum(m);
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
% figure
% scatter(pti(I, 1), pti(I, 2), 20, 'red')
% axis equal
% axis([0 1 0 1])

if grad_random == 0
    grad = [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, 2, m2);
end

V_atv1 = atv1(pti * W + b);
V_err1 = atv1(err_pt * W + b);
V_diff2_atv1 = sum(W .^ 2, 1) .* diff2_atv1(pti * W + b);

G = scale1 * sqrt(sum(grad .^ 2, 1));
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

% Ur(I2) = agrann.solve_output_layer(A(:, I2), F - A(:, I1) * agrann.solve_output_layer(A(:, I1), F, GPU, QR), GPU, QR);

Ur = agrann.solve_output_layer(A, F);

err_point = V_err1(:, I1) * Ur(I1) + atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)')) .* V_err2 * Ur(I2) - U_err;
err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

fprintf('L2_relative_error = %e, ', err_L2_rel);
fprintf('Linf_relative_error = %e\n', err_Linf_rel);

V = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)]  * W + b) * Ur_old * G - V .* G) * Ur(I2);

subplot(1, 4, 3)
agrann.visualization.draw_2d(X, Y, V)
subplot(1, 4, 4)
agrann.visualization.draw_2d(X, Y, abs(V - solu([X(:) Y(:)])))

print(agrann.figure_file('ex1_case3.eps'), '-depsc2')

toc

end
