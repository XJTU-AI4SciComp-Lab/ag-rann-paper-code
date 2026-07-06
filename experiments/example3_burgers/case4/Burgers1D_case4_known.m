function Burgers1D_case4_known

if nargin < 1
    num_inter = 200; 
    num_bound = 200;
    num_neuron = 1000;
    r_neuron = 15;
    dimension = 2;
    penalty_bound_init = 20;
    max_it = [10, 0, 2, 0];
end    

rng(1)

solu = @(x)agrann.problems.burgers_step_solution(x);
rhs = @(x)agrann.problems.zero_forcing(x);

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
r = r_neuron;
d = dimension;
eta0 = penalty_bound_init;

it_err = zeros(1, sum(max_it));
num_it_err = 1;

atv1 = @(x)tanh(x);
diff1_atv1 = @(x) 1 - tanh(x) .^ 2;

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, top, bottom);

W = unifrnd(-r, r, d, m);
b = [unifrnd(left, right, 1, m); unifrnd(bottom, top, 1, m)];
b = - sum(b .* W, 1);

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptt1 = linspace(left, right, (right - left) / (top - bottom) * nb)';
ptb = [[ptt1 bottom*ones(size(ptt1))];
       [zeros(nb, 1) linspace(bottom, top, nb)']];

U_err = solu(err_pt);
V_err = atv1(err_pt * W + b);

[X, Y] = meshgrid(linspace(left, right, 300), linspace(bottom, top, 500));

U_plt = solu([X(:) Y(:)]);
V_plt = atv1([X(:) Y(:)] * W + b);

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

    Ur = agrann.solve_output_layer(A, F);

    err_point = V_err * Ur - U_err;
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
    
    Ur = agrann.solve_output_layer(A, F);
    
    err_point = V_err * Ur - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

Draw2D_single(X, Y, solu([X(:) Y(:)]));

V = atv1([X(:) Y(:)] * W + b) * Ur;

Draw2D_single(X, Y, V);
Draw2D_single(X, Y, abs(V - solu([X(:) Y(:)])));

Ii1 = solu(pti) < 0.5;
Ii2 = solu(pti) > 0.5;
Ib1 = solu(ptb) < 0.5;
Ib2 = solu(ptb) > 0.5;
Ie1 = solu(err_pt) < 0.5;
Ie2 = solu(err_pt) > 0.5;
Ip1 = solu([X(:) Y(:)]) < 0.5;
Ip2 = solu([X(:) Y(:)]) > 0.5;

Ur = [Ur; Ur];

V_atv1 = atv1(pti * W + b);
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;
V_atv1 = blkdiag(V_atv1(Ii1, :), V_atv1(Ii2, :));
V_diff1_atv1_1 = blkdiag(V_diff1_atv1_1(Ii1, :), V_diff1_atv1_1(Ii2, :));
V_diff1_atv1_2 = blkdiag(V_diff1_atv1_2(Ii1, :), V_diff1_atv1_2(Ii2, :));

Ab = atv1(ptb * W + b);
Ab = blkdiag(Ab(Ib1, :), Ab(Ib2, :));
Fb = [solu(ptb(Ib1, :)); solu(ptb(Ib2, :))];

for i = 1 : max_it(3)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2;

    A = [Ai; eta0 * Ab];
    F = [Fi0; eta0 * Fb];

    Ur = agrann.solve_output_layer(A, F);

    V = - U_err;
    V(Ie1) = V(Ie1) + V_err(Ie1, :) * Ur(1 : m);
    V(Ie2) = V(Ie2) + V_err(Ie2, :) * Ur(m + 1: 2 * m);
    
    err_L2_rel = sqrt(err_wt' * V .^ 2 / (err_wt' * U_err .^ 2));
    fprintf('L2_relative_error = %e\n', err_L2_rel);  

    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

for i = 1 : max_it(4)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur);

    Fi = Fi0 + (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];
    
    Ur = agrann.solve_output_layer(A, F);
    
    V = - U_err;
    V(Ie1) = V(Ie1) + V_err(Ie1, :) * Ur(1 : m);
    V(Ie2) = V(Ie2) + V_err(Ie2, :) * Ur(m + 1: 2 * m);
    
    err_L2_rel = sqrt(err_wt' * V .^ 2 / (err_wt' * U_err .^ 2));
    fprintf('L2_relative_error = %e\n', err_L2_rel);  

    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

V = - U_plt;
V(Ip1) = V(Ip1) + V_plt(Ip1, :) * Ur(1 : m);
V(Ip2) = V(Ip2) + V_plt(Ip2, :) * Ur(m + 1: 2 * m);

gcf = figure('Units', 'centimeter', 'Position', [5 5 45 10]);
set(gcf, 'renderer', 'painters')

subplot(1, 3, 1)
Draw2D(X, Y, U_plt);
subplot(1, 3, 2)
Draw2D(X, Y, V + U_plt);
subplot(1, 3, 3)
Draw2D(X, Y, abs(V));

print(agrann.figure_file('ex3_case4_known.eps'), '-depsc2')

toc

end
