function Burgers1D_Growing_plot

if nargin < 1
    num_inter = 200; 
    num_bound = 300;
    num_neuron = 200 * ones(1, 5);
    r_neuron_range = 100;
    r_init = [10, 10];
    penalty_bound_init = 200;
    pre_weight = 100;
    E = 0.1 / pi;
    max_it = [10, 0, 5, 0];
end    

rng(1);

Fs = 100;

syms x y
u0 = - sin(pi * x);
G = matlabFunction(int(u0, x, 0, y));
u0 = matlabFunction(u0);
clear x y

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
r = r_neuron_range;
eta0 = penalty_bound_init;
pw = pre_weight;
res = zeros(3, length(m));

atv1 = @(x) exp(-x .^ 2 ./ 2);
diff1_atv1 = @(x) - x .* exp(- x .^ 2 / 2);
diff2_atv1 = @(x) x .^ 2 .* exp( - x .^ 2 / 2) - exp(- x .^ 2 / 2);

err_n = 100;
[err_pt, err_wt] = agrann.gauss_legendre_2d(err_n, left, right, bottom, top);
t0 = err_pt(:, 1);
x0 = err_pt(:, 2);
U_err = zeros(err_n ^ 2, 1);
for i = 1 : err_n ^ 2
    H = @(y)G(y) + (x0(i) - y) .^ 2 / 2 ./ t0(i);
    U_err(i) = integral(@(y)(x0(i) - y) ./ t0(i) .* exp(-H(y) / 2 / E), -inf, inf) / integral(@(y)exp(-H(y) / 2 / E), -inf, inf);
end

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptt1 = linspace(left, right, (right - left) / (top - bottom) * nb)';
ptbb = [[ptt1 bottom*ones(size(ptt1))];
       [ptt1 top*ones(size(ptt1))]];
ptbi = [zeros(nb, 1) linspace(bottom, top, nb)'];
ptb = [ptbb; ptbi];

T = 1 / Fs;
[tx, ty] = ndgrid(left : T : right, bottom : T : top);
L1 = length(left : T : right); 
L2 = length(bottom : T : top); 
[Fx, Fy] = ndgrid(Fs * (0 : (L1 / 2)) / L1, Fs * (0 : (L2 / 2)) / L2);

Wf = linspace(0, r, pw + 1);
Wf = Wf(2 : end);
[Wf1, Wf2] = meshgrid(Wf);
Wf = [Wf1(:)'; Wf2(:)'];

D2X = Wf(1, :) .* diff1_atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf) + Wf(2, :) .* diff1_atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf) .* (atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf)) - E * Wf(2, :) .^ 2 .* diff2_atv1([tx(:) - (left + right) / 2 ty(:) - (bottom + top) / 2] * Wf);
P_weight = abs(fft2(reshape(D2X, L1, L2, pw ^ 2)));
P_weight = P_weight(1 : ((L1 + 1) / 2), 1 : ((L2 + 1) / 2), :);

[~, maxPwt] = max(P_weight, [], [1, 2], 'linear'); 
maxPwt = reshape(maxPwt, 1, pw ^ 2) - (0 : pw ^ 2 - 1) * ((L1 + 1) / 2) * ((L2 + 1) / 2);
maxPwt = [maxPwt - floor(maxPwt / ((L1 + 1) / 2)) * ((L1 + 1) / 2); floor(maxPwt / ((L2 + 1) / 2)) + 1];

r = r_init;

W = [unifrnd(-r(1), r(1), 1, m(1)); unifrnd(-r(2), r(2), 1, m(1))];
b = [unifrnd(left, right, 1, m(1)); unifrnd(bottom, top, 1, m(1))];
b = - sum(b .* W, 1);

V_err1 = atv1([t0 x0] * W + b);

Ab = [atv1(ptbb * W + b); atv1(ptbi * W + b)];
Fb = [zeros(2 * size(ptt1, 1), 1); u0(ptbi(:, 2))];

Ur = zeros(m(1), 1);

V_atv1 = atv1(pti * W + b);
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;
V_diff2_atv1_2 = W(2, :) .^ 2 .* diff2_atv1(pti * W + b);

for j = 1 : max_it(1)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 - E * V_diff2_atv1_2;

    A = [Ai; eta0 * Ab];
    F = [zeros((right - left) / (top - bottom) * ni ^ 2, 1); eta0 * Fb];

    Ur = agrann.solve_output_layer(A, F);
    
    err_L2_rel =  sqrt(err_wt' * ((V_err1 * Ur - U_err) .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
    err_Linf_rel = max(abs(V_err1 * Ur - U_err)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);

end

for j = 1 : max_it(2)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur) - E * V_diff2_atv1_2;

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

X = reshape(atv1([tx(:) ty(:)] * W + b) * Ur, L1, L2);

t = tx(:);
x = ty(:);
u_real = zeros(L1 * L2, 1);
for i = 1 : L1 * L2
    H = @(y)G(y) + (x(i) - y) .^ 2 / 2 ./ t(i);
    u_real(i) = integral(@(y)(x(i) - y) ./ t(i) .* exp(-H(y) / 2 / E), -inf, inf) / integral(@(y)exp(-H(y) / 2 / E), -inf, inf);
end
u_real(1 : L1 : L1 * L2) = u0(x(1 : L1 : L1 * L2));

X_real = reshape(u_real, L1, L2);

X_lhs = reshape(W(1, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur + (W(2, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur) .* (atv1([tx(:) ty(:)] * W + b) * Ur) - E * W(2, :) .^ 2 .* diff2_atv1([tx(:) ty(:)] * W + b) * Ur, L1, L2);
Y_lhs = fft2(X_lhs);
P_lhs = abs(Y_lhs);
P_lhs = P_lhs(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

gcf = figure('Units', 'centimeter', 'Position', [5 5 33 12]);
set(gcf, 'renderer', 'painters')
annotation('textbox',[.02 .66 .2 .2],'String',{'   Time Domain', 'numerical solution'},'EdgeColor','none')
annotation('textbox',[.03 .365 .2 .2],'String',{'Time Domain', 'absolute error'},'EdgeColor','none')
annotation('textbox',[.02 .07 .2 .2],'String',{'Frequency Domain', '        residual'},'EdgeColor','none')

subplot(3, length(m), 1)
surf(tx, ty, X, EdgeColor = "none")
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
view(2); colorbar
axis square

subplot(3, length(m), length(m) + 1)
surf(tx, ty, abs(X - X_real), EdgeColor = "none")
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
view(2); colorbar
axis square

subplot(3, length(m), 2 * length(m) + 1)
surf(Fx(:, :), Fy(:, :), P_lhs(:, :), EdgeColor = "none")
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
view(2); colorbar
axis square
axis([0 30 0 30])

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
    V_diff2_atv1_2 = W(2, :) .^ 2 .* diff2_atv1(pti * W + b);

    for j = 1 : max_it(3)
        Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 - E * V_diff2_atv1_2;
    
        A = [Ai; eta0 * Ab];
        F = [zeros((right - left) / (top - bottom) * ni ^ 2, 1); eta0 * Fb];
    
        Ur = agrann.solve_output_layer(A, F);

        err_point = atv1([t0 x0] * W + b) * Ur - U_err;
        err_L2_rel =  sqrt(err_wt' * (err_point .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
        err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
        fprintf('L2_relative_error = %e, ', err_L2_rel);    
        fprintf('Linf_relative_error = %e\n', err_Linf_rel);

    end
    
    for j = 1 : max_it(4)
        Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur) - E * V_diff2_atv1_2;
    
        Fi = (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);
    
        A = [Ai; eta0 * Ab];
        F = [Fi; eta0 * Fb];
    
        Ur = agrann.solve_output_layer(A, F);
    
        err_point = atv1([t0 x0] * W + b) * Ur - U_err;
        err_L2_rel =  sqrt(err_wt' * (err_point .^ 2)) / sqrt(err_wt' * (U_err .^ 2));
        err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
        
        fprintf('L2_relative_error = %e, ', err_L2_rel);    
        fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    
    end

    res(1:2, i) = r;
    res(3, i) = err_L2_rel;

    X = reshape(atv1([tx(:) ty(:)] * W + b) * Ur, L1, L2);

    X_lhs = reshape(W(1, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur + (W(2, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur) .* (atv1([tx(:) ty(:)] * W + b) * Ur) - E * W(2, :) .^ 2 .* diff2_atv1([tx(:) ty(:)] * W + b) * Ur, L1, L2);
    Y_lhs = fft2(X_lhs);
    P_lhs = abs(Y_lhs);
    P_lhs = P_lhs(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

    subplot(3, length(m), i)
    surf(tx, ty, X, EdgeColor = "none")
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
    view(2); colorbar
    axis square

    subplot(3, length(m), length(m) + i)
    surf(tx, ty, abs(X - X_real), EdgeColor = "none")
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
    view(2); colorbar
    axis square

    subplot(3, length(m), 2 * length(m) + i)
    surf(Fx(:, :), Fy(:, :), P_lhs(:, :), EdgeColor = "none")
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
    view(2); colorbar
    axis square
    axis([0 30 0 30])

end

print(agrann.figure_file('ex3_case1_Growing.eps'), '-depsc2')

for i = 1 : length(m)
    fprintf('(%.2f,%.2f) & ', res(1, i), res(2, i))
end
fprintf('\n');
for i = 1 : length(m)
    fprintf('%.4e & ', res(3, i))
end
fprintf('\n');

toc

end
