function Poisson1D_Best

if nargin < 1
    num_inter = 2000; 
    num_neuron = 500;
    r_neuron = 200;
    penalty_bound_init = 1;
    k = 6;
end

rng(1);

Fs = 10000;
d = 250;

solu = @(x)agrann.problems.poisson_multiscale_1d_solution(x, k);
rhs = @(x)agrann.problems.poisson_multiscale_1d_forcing(x, k);
left = 0;
right = 1;

close all
format long e
warning off

tic

ni = num_inter; 
m = num_neuron;
r = r_neuron;
eta0 = penalty_bound_init;

% atv1 = @(x)tanh(x);
% diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

atv1 = @(x) exp(-x .^ 2 ./ 2);
diff2_atv1 = @(x) x .^ 2 .* exp( - x .^ 2 / 2) - exp(- x .^ 2 / 2);

[err_pt, err_wt] = agrann.gauss_legendre_1d(200, left, right);

eps = 1e-10;

pti = linspace(left + eps, right - eps, ni)';
ptb = [left; right];

U_err = solu(err_pt);

W = unifrnd(-r, r, 1, m);
b = unifrnd(left, right, 1, m);
b = - sum(b .* W, 1);

Ai = - W .^ 2 .* diff2_atv1(pti * W + b);
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F);

err_L2_rel = sqrt(err_wt' * (atv1(err_pt * W + b) * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);   

T = 1 / Fs;
t = (left : T : right)';
L = length(t); 
fs = Fs * (0 : (L / 2)) / L;

X = atv1(t * W + b) * Ur;
Y = fft(X);
P = abs(Y);

X_real = solu(t);
Y_real = fft(X_real);
P_real = abs(Y_real);

X_lhs = - W .^ 2 .* diff2_atv1(t * W + b) * Ur;
Y_lhs = fft(X_lhs);
P_lhs = abs(Y_lhs);

P_rhs = abs(fft(rhs(t)));
P_rhs(Fs / 2 + 2 : end) = [];

gcf = figure('Units', 'centimeter', 'Position', [5 5 33 15]);
set(gcf, 'renderer', 'painters')
annotation('textbox',[.04 .65 .1 .2],'String',{'  Time','Domain'},'EdgeColor','none')
annotation('textbox',[.03 .35 .1 .2],'String',{'Frequency','  Domain'},'EdgeColor','none')
annotation('textbox',[.03 .05 .1 .2],'String',{'Frequency','  Domain'},'EdgeColor','none')

subplot(3, 3, 1)
plot(t, X);
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("numerical solution", FontSize=12)

subplot(3, 3, 2)
plot(t, X_real);
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("exact solution", FontSize=12)

subplot(3, 3, 3)
plot(t, abs(X - X_real))
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("absolute error", FontSize=12)

subplot(3, 3, 4)
plot(fs(1:d), P(1:d)) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("numerical solution", FontSize=12)

subplot(3, 3, 5)
plot(fs(1:d), P_real(1:d)) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("exact solution", FontSize=12)

subplot(3, 3, 6)
plot(fs(1:d), abs(P(1:d) - P_real(1:d)))
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("absolute error", FontSize=12)

subplot(3, 3, 7)
plot(fs(1:d), P_lhs(1:d)) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("left-hand-side", FontSize=12)

subplot(3, 3, 8)
plot(fs(1:d), P_rhs(1:d)) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("right-hand-side", FontSize=12)

subplot(3, 3, 9)
plot(fs(1:d), abs(P_lhs(1:d) - P_rhs(1:d))) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
title("residual", FontSize=12)

print(agrann.figure_file('ex1_case1_best.eps'), '-depsc2')

toc

end
