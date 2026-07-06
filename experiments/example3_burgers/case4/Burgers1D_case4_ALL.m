function Burgers1D_case4_ALL

if nargin < 1
    num_inter = 300; 
    num_bound = 400;
    num_neuron = 100 * ones(1, 4);
    num_neuron_add = 700;
    r_neuron_range = 100;
    r_init = [1, 1];
    penalty_bound_init = 20;
    pre_weight = 100;
    scale1 = 1;
    scale2 = 1;
    grad_random = 10;
    max_it = [10, 0, 5, 0, 0, 5, 2, 0];
end    

rng(1);

Fs = 100;

solu = @(x)agrann.problems.burgers_step_solution(x);

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
r = r_neuron_range;
eta0 = penalty_bound_init;
pw = pre_weight;
res = zeros(3, length(m));

atv1 = @(x) exp(-x .^ 2 ./ 2);
diff1_atv1 = @(x) - x .* exp(- x .^ 2 / 2);

[atv0, diff1_atv0, ~] = agrann.wavelet_basis(0);
[atv21, diff1_atv21, ~] = agrann.wavelet_basis(0);

atv22 = @(x) (sign(x) + 1) / 2 .* x + (sign(x) + 1) / 2;
diff1_atv22 = @(x) (sign(x) + 1) / 2;

atv2 = atv21;

err_n = 100;
[err_pt, err_wt] = agrann.gauss_legendre_2d(err_n, left, right, bottom, top);

U_err = solu(err_pt);

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptt1 = linspace(left, right, (right - left) / (top - bottom) * nb)';
ptbb = [ptt1 bottom*ones(size(ptt1))];
ptbi = [zeros(nb, 1) linspace(bottom, top, nb)'];
ptb = [ptbb; ptbi];

T = 1 / Fs;
[tx, ty] = ndgrid(left : T : right, bottom : T : top);
L1 = length(left : T : right); 
L2 = length(bottom : T : top); 

Wf = linspace(0, r, pw + 1);
Wf = Wf(2 : end);
[Wf1, Wf2] = meshgrid(Wf);
Wf = [Wf1(:)'; Wf2(:)'];

D2X = Wf(1, :) .* diff1_atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf) + Wf(2, :) .* diff1_atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf) .* (atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf));
P_weight = abs(fft2(reshape(D2X, L1, L2, pw ^ 2)));
P_weight = P_weight(1 : ((L1 + 1) / 2), 1 : ((L2 + 1) / 2), :);

[~, maxPwt] = max(P_weight, [], [1, 2], 'linear'); 
maxPwt = reshape(maxPwt, 1, pw ^ 2) - (0 : pw ^ 2 - 1) * ((L1 + 1) / 2) * ((L2 + 1) / 2);
maxPwt = [maxPwt - floor(maxPwt / ((L1 + 1) / 2)) * ((L1 + 1) / 2); floor(maxPwt / ((L2 + 1) / 2)) + 1];

r = r_init;

W = [unifrnd(-r(1), r(1), 1, m(1)); unifrnd(-r(2), r(2), 1, m(1))];
b = [unifrnd(left, right, 1, m(1)); unifrnd(bottom, top, 1, m(1))];
b = - sum(b .* W, 1);

V_err1 = atv1(err_pt * W + b);

Ab = [atv1(ptbb * W + b); atv1(ptbi * W + b)];
Fb = solu(ptb);

Ur = zeros(m(1), 1);

V_atv1 = atv1(pti * W + b);
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;

for j = 1 : max_it(1)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2;

    A = [Ai; eta0 * Ab];
    F = [zeros(size(pti, 1), 1); eta0 * Fb];

    Ur = agrann.solve_output_layer(A, F);
    
    err_L2_rel =  sqrt(err_wt' * ((V_err1 * Ur - U_err) .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
    err_Linf_rel = max(abs(V_err1 * Ur - U_err)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

end

for j = 1 : max_it(2)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur);

    Fi = (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur = agrann.solve_output_layer(A, F);

    err_point = V_err1 * Ur - U_err;
    err_L2_rel =  sqrt(err_wt' * (err_point .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

end

res(1:2, 1) = r;
res(3, 1) = err_L2_rel;

OLD = atv1(err_pt * W + b) * Ur;

X_lhs = reshape(W(1, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur + (W(2, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur) .* (atv1([tx(:) ty(:)] * W + b) * Ur), L1, L2);
Y_lhs = fft2(X_lhs);
P_lhs = abs(Y_lhs);
P_lhs = P_lhs(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

for i = 2 : length(m)
    [~, maxrhs] = max(abs(P_lhs), [], "all"); 
    maxrhs = [maxrhs - floor(maxrhs / ((L1 + 1) / 2)) * ((L1 + 1) / 2); floor(maxrhs / ((L2 + 1) / 2)) + 1];
    I = [find(maxPwt(1, :) > maxrhs(1) & maxPwt(2, :) > maxrhs(2)) pw ^ 2];
    [~, r] = min(sum(maxPwt(:, I) .^ 2, 1));
    r = Wf(:, I(r));
    fprintf('r = (%.6f, %.6f)\n', r(1), r(2))

    Ur = [Ur; zeros(m(i), 1)];

    W_add = [unifrnd(-r(1), r(1), 1, m(i)); unifrnd(-r(2), r(2), 1, m(i))];
    b_add = unifrnd(left, right, 2, m(i));
    b_add = - sum(b_add .* W_add, 1);
    W = [W, W_add];
    b = [b, b_add];

    Ab = [Ab, atv1(ptb * W_add + b_add)];

    V_atv1 = atv1(pti * W + b);
    V_diff1_atv1 = diff1_atv1(pti * W + b);
    V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
    V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;

    for j = 1 : max_it(3)
        Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2;
    
        A = [Ai; eta0 * Ab];
        F = [zeros(size(pti, 1), 1); eta0 * Fb];
    
        Ur = agrann.solve_output_layer(A, F);

        err_point = atv1(err_pt * W + b) * Ur - U_err;
        err_L2_rel =  sqrt(err_wt' * (err_point .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
        err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
        fprintf('L2_relative_error = %e, ', err_L2_rel);    
        fprintf('Linf_relative_error = %e\n', err_Linf_rel);

    end
    
    for j = 1 : max_it(4)
        Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur);
    
        Fi = (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);
    
        A = [Ai; eta0 * Ab];
        F = [Fi; eta0 * Fb];
    
        Ur = agrann.solve_output_layer(A, F);
    
        err_point = atv1(err_pt * W + b) * Ur - U_err;
        err_L2_rel =  sqrt(err_wt' * (err_point .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
        err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
        
        fprintf('L2_relative_error = %e, ', err_L2_rel);    
        fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    
    end

    res(1:2, i) = r;
    res(3, i) = err_L2_rel;

    X_lhs = reshape(W(1, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur + (W(2, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur) .* (atv1([tx(:) ty(:)] * W + b) * Ur), L1, L2);
    Y_lhs = fft2(X_lhs);
    P_lhs = abs(Y_lhs);
    P_lhs = P_lhs(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

    fprintf('%.4e\n', abs(err_wt' * (OLD - atv1(err_pt * W + b) * Ur)))

    if abs(err_wt' * (OLD - atv1(err_pt * W + b) * Ur)) < 5e-4
        break
    end
    OLD = atv1(err_pt * W + b) * Ur;
end

for j = 1 : i
    fprintf('(%.2f,%.2f) & ', res(1, j), res(2, j))
end
fprintf('\n');
for j = 1 : i
    fprintf('%.4e & ', res(3, j))
end
fprintf('\n');

[X, Y] = meshgrid(linspace(left, right, 501), linspace(bottom, top, 501));

gcf = figure('Position', [0 0 1500 1200]);
% set(gcf, 'renderer', 'painters')

V = atv1([X(:) Y(:)] * W + b) * Ur;
fprintf('Linf_relative_error = %e\n', max(abs(V - solu([X(:) Y(:)]))));

subplot(4, 4, 1)
Draw2D(X, Y, V);
subplot(4, 4, 2)
Draw2D_3(X, Y, V);
subplot(4, 4, 3)
Draw1D_1(X, Y, V, solu);
subplot(4, 4, 4)
Draw1D_2(X, Y, V, solu);

m = sum(m(1:i));

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

if grad_random == 0
    grad = scale1 * [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, 2, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (atv1(pti(I, :) * W + b) * Ur)';

grad = scale2 * grad;

Ab(:, I2) = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv21(atv1(ptb * W + b) * Ur * G - V .* G);

Ur_new = [Ur; zeros(m - size(Ai, 2), 1)];

V_diff1_atv1_0 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1_0;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1_0;

V_atv0_1 = atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_atv0_2 = atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));
V_diff1_atv0_1 = grad(1, :) .* diff1_atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_diff1_atv0_2 = grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));

V_atv2_0 = V_atv1 * Ur * G - V .* G;
V_atv2 = atv21(V_atv2_0);
V_diff1_atv2_0 = diff1_atv21(V_atv2_0) .* G;
V_diff1_atv2_1 = V_diff1_atv2_0 .* (V_diff1_atv1_1 * Ur);
V_diff1_atv2_2 = V_diff1_atv2_0 .* (V_diff1_atv1_2 * Ur);

V_err0_1 = atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)'));
V_err0_2 = atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)'));
V_err1 = atv1(err_pt * W + b);
V_err2 = atv21(V_err1 * Ur * G - V .* G);

for i = 1 : max_it(5)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Ai = zeros(size(pti, 1), m);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1);

    Ai(:, I2) = V_atv0_2 .* (V_diff1_atv0_1 .* V_atv2 + V_atv0_1 .* V_diff1_atv2_1) + ...
          Un .* V_atv0_1 .* (V_diff1_atv0_2 .* V_atv2 + V_atv0_2 .* V_diff1_atv2_2);

    A = [Ai; eta0 * Ab];
    F = [zeros(size(pti, 1), 1); eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
end

for i = 1 : max_it(6)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Unx = V_diff1_atv1_2(:, I1) * Ur_new(I1) + V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 * Ur_new(I2) + ...
                                               V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2 * Ur_new(I2);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1) + V_atv1(:, I1) .* Unx;

    Ai(:, I2) = V_diff1_atv0_1 .* V_atv0_2 .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_1 + ...
          Un .* (V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2) + ...
          V_atv0_1 .* V_atv0_2 .* V_atv2 .* Unx;

    Fi = Un .* Unx;

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
end

Vp = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur_new(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv21(atv1([X(:) Y(:)]  * W + b) * Ur * G - V .* G) * Ur_new(I2);

fprintf('Linf_relative_error = %e\n', max(abs(Vp - solu([X(:) Y(:)]))));

Ur_new0 = Ur_new;

subplot(4, 4, 5)
Draw2D(X, Y, Vp);
subplot(4, 4, 6)
Draw2D_3(X, Y, Vp);
subplot(4, 4, 7)
Draw1D_1(X, Y, Vp, solu);
subplot(4, 4, 8)
Draw1D_2(X, Y, Vp, solu);

Ab(:, I2) = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv22(atv1(ptb * W + b) * Ur * G - V .* G);
V_atv2 = atv22(V_atv2_0);
V_diff1_atv2_0 = diff1_atv22(V_atv2_0) .* G;
V_diff1_atv2_1 = V_diff1_atv2_0 .* (V_diff1_atv1_1 * Ur);
V_diff1_atv2_2 = V_diff1_atv2_0 .* (V_diff1_atv1_2 * Ur);
V_err2 = atv22(V_err1 * Ur * G - V .* G);
Ur_new = [Ur; zeros(m + m2 - 1000, 1)];

for i = 1 : max_it(5)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Ai = zeros(size(pti, 1), m);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1);

    Ai(:, I2) = V_atv0_2 .* (V_diff1_atv0_1 .* V_atv2 + V_atv0_1 .* V_diff1_atv2_1) + ...
          Un .* V_atv0_1 .* (V_diff1_atv0_2 .* V_atv2 + V_atv0_2 .* V_diff1_atv2_2);

    A = [Ai; eta0 * Ab];
    F = [zeros(size(pti, 1), 1); eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
end

for i = 1 : max_it(6)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Unx = V_diff1_atv1_2(:, I1) * Ur_new(I1) + V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 * Ur_new(I2) + ...
                                               V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2 * Ur_new(I2);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1) + V_atv1(:, I1) .* Unx;

    Ai(:, I2) = V_diff1_atv0_1 .* V_atv0_2 .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_1 + ...
          Un .* (V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2) + ...
          V_atv0_1 .* V_atv0_2 .* V_atv2 .* Unx;

    Fi = Un .* Unx;

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
end

Vp = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur_new(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv22(atv1([X(:) Y(:)]  * W + b) * Ur * G - V .* G) * Ur_new(I2);

fprintf('Linf_relative_error = %e\n', max(abs(Vp - solu([X(:) Y(:)]))));

subplot(4, 4, 9)
Draw2D(X, Y, Vp);
subplot(4, 4, 10)
Draw2D_3(X, Y, Vp);
subplot(4, 4, 11)
Draw1D_1(X, Y, Vp, solu);
subplot(4, 4, 12)
Draw1D_2(X, Y, Vp, solu);

mid = err_point + U_err;
mid = (max(mid) + min(mid)) / 2;

V_err1 = atv1(pti * W + b);
TTTT = V_err1(:, I1) * Ur_new0(I1) + atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')) .* atv2(V_err1 * Ur * G - V .* G) * Ur_new0(I2);
Ii1 = TTTT < mid;
Ii2 = TTTT > mid;

V_err1 = atv1(ptb * W + b);
TTTT = V_err1(:, I1) * Ur_new0(I1) + atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv2(V_err1 * Ur * G - V .* G) * Ur_new0(I2);
Ib1 = TTTT < mid;
Ib2 = TTTT > mid;

V_err1 = atv1(err_pt * W + b);
TTTT = V_err1(:, I1) * Ur_new0(I1) + atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)')) .* atv2(V_err1 * Ur * G - V .* G) * Ur_new0(I2);
Ie1 = TTTT < mid;
Ie2 = TTTT > mid;

V_err1 = atv1([X(:) Y(:)] * W + b);
TTTT = V_err1(:, I1) * Ur_new0(I1) + atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(V_err1 * Ur * G - V .* G) * Ur_new0(I2);
Ip1 = TTTT < mid;
Ip2 = TTTT > mid;

m = 500;
Ur = zeros(2 * m, 1);

r = 2;
W = unifrnd(-r, r, 2, m);
b = [unifrnd(left, right, 1, m); unifrnd(bottom, top, 1, m)];
b = - sum(b .* W, 1);

V_plt = atv1([X(:) Y(:)] * W + b);

V_atv1 = atv1(pti * W + b);
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;
V_atv1 = blkdiag(V_atv1(Ii1, :), V_atv1(Ii2, :));
V_diff1_atv1_1 = blkdiag(V_diff1_atv1_1(Ii1, :), V_diff1_atv1_1(Ii2, :));
V_diff1_atv1_2 = blkdiag(V_diff1_atv1_2(Ii1, :), V_diff1_atv1_2(Ii2, :));

U_err = solu(err_pt);
V_err = atv1(err_pt * W + b);

Ab = atv1(ptb * W + b);
Ab = blkdiag(Ab(Ib1, :), Ab(Ib2, :));
Fb = [solu(ptb(Ib1, :)); solu(ptb(Ib2, :))];

for i = 1 : max_it(7)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2;

    A = [Ai; eta0 * Ab];
    F = [zeros(size(pti, 1), 1); eta0 * Fb];

    Ur = agrann.solve_output_layer(A, F);

    V = - U_err;
    V(Ie1) = V(Ie1) + V_err(Ie1, :) * Ur(1 : m);
    V(Ie2) = V(Ie2) + V_err(Ie2, :) * Ur(m + 1: 2 * m);

    err_L2_rel = sqrt(err_wt' * V .^ 2 / (err_wt' * U_err .^ 2));
    fprintf('L2_relative_error = %e\n', err_L2_rel);  
end

for i = 1 : max_it(8)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur);

    Fi = (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur = agrann.solve_output_layer(A, F);

    V = - U_err;
    V(Ie1) = V(Ie1) + V_err(Ie1, :) * Ur(1 : m);
    V(Ie2) = V(Ie2) + V_err(Ie2, :) * Ur(m + 1: 2 * m);

    err_L2_rel = sqrt(err_wt' * V .^ 2 / (err_wt' * U_err .^ 2));
    fprintf('L2_relative_error = %e\n', err_L2_rel);  
end

V(Ip1) = V_plt(Ip1, :) * Ur(1 : m);
V(Ip2) = V_plt(Ip2, :) * Ur(m + 1: 2 * m);

fprintf('Linf_relative_error = %e\n', max(abs(V - solu([X(:) Y(:)]))));

subplot(4, 4, 13)
Draw2D(X, Y, V);
subplot(4, 4, 14)
Draw2D_3(X, Y, V);
subplot(4, 4, 15);
Draw1D_1(X, Y, V, solu);
subplot(4, 4, 16)
Draw1D_2(X, Y, V, solu);

print(agrann.figure_file('ex3_case4_all.eps'), '-depsc')

toc

end
