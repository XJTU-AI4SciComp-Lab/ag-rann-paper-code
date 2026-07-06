function Burgers1D_Growing_best

if nargin < 1
    num_inter = 200; 
    num_bound = 300;
    num_neuron = 1000 * ones(1, 1);
    r_neuron_range = 100;
    r_init = [8, 25];
    penalty_bound_init = 100;
    pre_weight = 100;
    E = 0.1 / pi;
    max_it = [20, 0, 0, 0];
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
eta0 = penalty_bound_init;

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

T = 1 / Fs;
[tx, ty] = ndgrid(left : T : right, bottom : T : top);
L1 = length(left : T : right); 
L2 = length(bottom : T : top); 
[Fx, Fy] = ndgrid(Fs * (0 : (L1 / 2)) / L1, Fs * (0 : (L2 / 2)) / L2);

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

X = reshape(atv1([tx(:) ty(:)] * W + b) * Ur, L1, L2);
Y = fft2(X);
P = abs(Y);
P = P(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

t = tx(:);
x = ty(:);
u_real = zeros(L1 * L2, 1);
for i = 1 : L1 * L2
    H = @(y)G(y) + (x(i) - y) .^ 2 / 2 ./ t(i);
    u_real(i) = integral(@(y)(x(i) - y) ./ t(i) .* exp(-H(y) / 2 / E), -inf, inf) / integral(@(y)exp(-H(y) / 2 / E), -inf, inf);
end
u_real(1 : L1 : L1 * L2) = u0(x(1 : L1 : L1 * L2));

X_real = reshape(u_real, L1, L2);
Y_real = fft2(X_real);
P_real = abs(Y_real);
P_real = P_real(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

X_lhs = reshape(W(1, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur + (W(2, :) .* diff1_atv1([tx(:) ty(:)] * W + b) * Ur) .* (atv1([tx(:) ty(:)] * W + b) * Ur) - E * W(2, :) .^ 2 .* diff2_atv1([tx(:) ty(:)] * W + b) * Ur, L1, L2);
Y_lhs = fft2(X_lhs);
P_lhs = abs(Y_lhs);
P_lhs = P_lhs(1 : (L1 + 1) / 2, 1 : (L2 + 1) / 2);

gcf = figure('Units', 'centimeter', 'Position', [5 5 25 15]);
set(gcf, 'renderer', 'painters')
annotation('textbox',[.04 .65 .1 .2],'String',{'  Time','Domain'},'EdgeColor','none')
annotation('textbox',[.03 .35 .1 .2],'String',{'Frequency','  Domain'},'EdgeColor','none')
annotation('textbox',[.03 .05 .1 .2],'String',{'Frequency','  Domain'},'EdgeColor','none')

subplot(3, 3, 1)
surf(tx, ty, X, EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
axis([left right bottom top])
title("numerical solution", FontSize=12)

subplot(3, 3, 2)
surf(tx, ty, X_real, EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
axis([left right bottom top])
title("exact solution", FontSize=12)

subplot(3, 3, 3)
surf(tx, ty, abs(X - X_real), EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
axis([left right bottom top])
title("absolute error", FontSize=12)

subplot(3, 3, 4)
surf(Fx, Fy, P, EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 20 0 20])
title("numerical solution", FontSize=12)

subplot(3, 3, 5)
surf(Fx, Fy, P_real, EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 20 0 20])
title("exact solution", FontSize=12)

subplot(3, 3, 6)
surf(Fx, Fy, abs(P - P_real), EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 20 0 20])
title("absolute error", FontSize=12)

subplot(3, 3, 7)
surf(Fx, Fy, P_lhs, EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 20 0 20])
title("left-hand-side", FontSize=12)

subplot(3, 3, 8)
surf(Fx, Fy, zeros(size(Fx)), EdgeColor = "none")
view(2); colorbar
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 20 0 20])
title("right-hand-side", FontSize=12)

subplot(3, 3, 9)
surf(Fx, Fy, abs(P_lhs), EdgeColor = "none")
view(2); colorbar 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 20 0 20])
title("residual", FontSize=12)

print(agrann.figure_file('ex3_case1_best.eps'), '-depsc2')

toc

end
