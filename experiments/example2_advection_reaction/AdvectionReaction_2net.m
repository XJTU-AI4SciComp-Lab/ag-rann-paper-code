function AdvectionReaction_2net

if nargin < 1
    num_inter = 100; 
    num_bound = 200;
    num_neuron = 1000;
    r_neuron = [20, 1];
    dimension = 2;
    penalty_bound_init = 1;
end    

rng(1)

solu = @(x)agrann.problems.advection_reaction_step_solution(x);
rhs = @(x)agrann.problems.advection_reaction_step_forcing(x);

GPU = 0;
QR = 1;

left = 0;
right = 1;
bottom = 0;
top = 1;

close all
format long e
warning off

tic

ni = num_inter; 
nb = num_bound;
m = num_neuron;
r = r_neuron;
eta0 = penalty_bound_init;

% atv1 = @(x)tanh(x);
% diff1_atv1 = @(x) 1 - tanh(x) .^ 2;

atv1 = @(x) exp(-x .^ 2 ./ 2);
diff1_atv1 = @(x) -x .* exp(-x .^ 2 / 2);

W = [unifrnd(-r(1), r(1), 1, m); unifrnd(-r(2), r(2), 1, m)];
b = [unifrnd(left, right, 1, m); unifrnd(bottom, top, 1, m)];
b = - sum(b .* W, 1);

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptt1 = linspace(left, right, nb)';
ptb = [ptt1 bottom*ones(nb, 1)];

plt_n = 500;
ptt1 = linspace(left, right, plt_n);
ptt2 = linspace(bottom, top, plt_n);
[pttx, ptty] = meshgrid(ptt1, ptt2);
plt_pt = [pttx(:) ptty(:)];

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, left, right);

U_plt = solu(plt_pt);
V_plt = atv1(plt_pt * W + b);

U_err = solu(err_pt);
V_err = atv1(err_pt * W + b);

Ai = W(2, :) .* diff1_atv1(pti * W + b) + atv1(pti * W + b);
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F, GPU, QR);

err_L2_rel = sqrt(err_wt' * (V_err * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);    

gcf = figure('Units', 'centimeter', 'Position', [5 5 36 10]);
set(gcf, 'renderer', 'painters')
subplot(1, 3, 1)
Draw2D_single(pttx, ptty, solu([pttx(:) ptty(:)]));

V = atv1([pttx(:) ptty(:)] * W + b) * Ur;
subplot(1, 3, 2)
Draw2D_single(pttx, ptty, V);
% print('ex2_case2_first.eps', '-depsc2')

mid = atv1(pti * W + b) * Ur;
mid = (max(mid) + min(mid)) / 2;

Ii1 = atv1(pti * W + b) * Ur < mid;
Ii2 = atv1(pti * W + b) * Ur > mid;
Ib1 = atv1(ptb * W + b) * Ur < mid;
Ib2 = atv1(ptb * W + b) * Ur > mid;
Ie1 = atv1(err_pt * W + b) * Ur < mid;
Ie2 = atv1(err_pt * W + b) * Ur > mid;
Ip1 = atv1(plt_pt * W + b) * Ur < mid;
Ip2 = atv1(plt_pt * W + b) * Ur > mid;

subfunc(W, b, pti, ptb, m, atv1, diff1_atv1, solu, rhs, eta0, U_plt, V_plt, U_err, V_err, pttx, ptty, err_wt, Ii1, Ii2, Ib1, Ib2, Ie1, Ie2, Ip1, Ip2)
print(agrann.figure_file('ex2_case2_unknown.eps'), '-depsc2')

toc

end

function subfunc(W, b, pti, ptb, m, atv1, diff1_atv1, solu, rhs, eta0, U_plt, V_plt, U_err, V_err, pttx, ptty, err_wt, Ii1, Ii2, Ib1, Ib2, Ie1, Ie2, Ip1, Ip2)

Ai = W(2, :) .* diff1_atv1(pti * W + b) + atv1(pti * W + b);
Ai = blkdiag(Ai(Ii1, :), Ai(Ii2, :));
Fi = [rhs(pti(Ii1, :)); rhs(pti(Ii2, :))];

Ab = atv1(ptb * W + b);
Ab = blkdiag(Ab(Ib1, :), Ab(Ib2, :));
Fb = [solu(ptb(Ib1, :)); solu(ptb(Ib2, :))];

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F);

V = - U_plt;
V(Ip1) = V(Ip1) + V_plt(Ip1, :) * Ur(1 : m);
V(Ip2) = V(Ip2) + V_plt(Ip2, :) * Ur(m + 1: 2 * m);

subplot(1, 3, 3)
Draw2D_single(pttx, ptty, V + U_plt);

V = - U_err;
V(Ie1) = V(Ie1) + V_err(Ie1, :) * Ur(1 : m);
V(Ie2) = V(Ie2) + V_err(Ie2, :) * Ur(m + 1: 2 * m);

err_L2_rel = sqrt(err_wt' * V .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);  
end
