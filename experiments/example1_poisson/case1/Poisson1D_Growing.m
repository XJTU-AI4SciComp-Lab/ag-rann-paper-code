function Poisson1D_Growing

if nargin < 1
    num_inter = 2000; 
    num_neuron = 50 * [4 2 1 1 1 1];
    r_neuron_range = 800;
    penalty_bound_init = 1;
    k = 6;
    pre_weight = 100;
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
r = r_neuron_range;
eta0 = penalty_bound_init;
pw = pre_weight;
res = zeros(2, length(m));

% atv1 = @(x)tanh(x);
% diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

atv1 = @(x) exp(-x .^ 2 ./ 2);
diff2_atv1 = @(x) x .^ 2 .* exp( - x .^ 2 / 2) - exp(- x .^ 2 / 2);

[err_pt, err_wt] = agrann.gauss_legendre_1d(200, left, right);

eps = 1e-10;

pti = linspace(left + eps, right - eps, ni)';
ptb = [left; right];

T = 1 / Fs;
t = (left : T : right)';
L = length(t); 
fs = Fs * (0 : (L / 2)) / L;

P_rhs = abs(fft(rhs(t)));
P_rhs(Fs / 2 + 2 : end) = [];

% plot(fs(1:d), P_rhs(1:d)) 

Wf = linspace(0, r, pw + 1);
Wf = Wf(2 : end);
D2X = - Wf .^ 2 .* diff2_atv1((t - (left + right) / 2) * Wf);
P_weight = abs(fft(D2X));

[~, maxrhs] = max(P_rhs); 
[~, maxPwt] = max(P_weight); 
r = find(maxPwt > maxrhs);
r = Wf(r(1));
fprintf('r = %.6f\n', r)

% figure
% for i = 1 : pw
%     nexttile
%     hold on
%     plot(fs(1:d), P_weight(1:d, i), LineWidth=2) 
%     scatter(fs(maxPwt(i)), P_weight(maxPwt(i), i), 'filled')
%     title(['r = ', num2str(Wf(i))])
% end

U_err = solu(err_pt);

W = unifrnd(-r, r, 1, m(1));
b = unifrnd(left, right, 1, m(1));
b = - sum(b .* W, 1);

Ai = - W .^ 2 .* diff2_atv1(pti * W + b);
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur = agrann.solve_output_layer(A, F);

err_L2_rel =  sqrt(err_wt' * (atv1(err_pt * W + b) * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);    
res(1, 1) = r;
res(2, 1) = err_L2_rel;

X = atv1(t * W + b) * Ur;

X_real = solu(t);

X_lhs = - W .^ 2 .* diff2_atv1(t * W + b) * Ur;
Y_lhs = fft(X_lhs);
P_lhs = abs(Y_lhs);

gcf = figure('Units', 'centimeter', 'Position', [5 5 33 15]);
set(gcf, 'renderer', 'painters')
annotation('textbox',[.02 .67 .2 .2],'String',{'   Time Domain', 'numerical solution'},'EdgeColor','none')
annotation('textbox',[.03 .45 .2 .2],'String',{'Time Domain', 'absolute error'},'EdgeColor','none')
annotation('textbox',[.02 .25 .2 .2],'String',{'Frequency Domain', '    left-hand-side'},'EdgeColor','none')
annotation('textbox',[.02 .02 .2 .2],'String',{'Frequency Domain', '        residual'},'EdgeColor','none')
 
subplot(4, length(m), 1)
plot(t, X);
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

subplot(4, length(m), length(m) + 1)
plot(t, X - X_real)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

subplot(4, length(m), 2 * length(m) + 1)
plot(fs(1:d), P_lhs(1:d)) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

subplot(4, length(m), 3 * length(m) + 1)
plot(fs(1:d), abs(P_lhs(1:d) - P_rhs(1:d))) 
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

for i = 2 : length(m)
    % [~, maxrhs] = max(abs(P_lhs(I) - P_rhs(I))); 
    % r = maxPwt([find(maxPwt > I(maxrhs)) pw]);
    P_lhs(Fs / 2 + 2 : end) = [];
    
    [~, maxrhs] = max(abs(P_lhs - P_rhs)); 
    r = [find(maxPwt > maxrhs) pw];
    r = Wf(r(1));
    fprintf('r = %.6f\n', r)

    W_add = unifrnd(-r, r, 1, m(i));
    b_add = unifrnd(left, right, 1, m(i));
    b_add = - sum(b_add .* W_add, 1);
    W = [W, W_add];
    b = [b, b_add];

    Ai = [Ai, - W_add .^ 2 .* diff2_atv1(pti * W_add + b_add)];
    
    Ab = [Ab, atv1(ptb * W_add + b_add)];
    
    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];
    
    Ur = agrann.solve_output_layer(A, F);
    
    err_L2_rel =  sqrt(err_wt' * (atv1(err_pt * W + b) * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
    fprintf('L2_relative_error = %e\n', err_L2_rel);  
    res(1, i) = r;
    res(2, i) = err_L2_rel;

    X = atv1(t * W + b) * Ur;
    
    X_lhs = - W .^ 2 .* diff2_atv1(t * W + b) * Ur;
    Y_lhs = fft(X_lhs);
    P_lhs = abs(Y_lhs);

    subplot(4, length(m), i)
    plot(t, X);
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

    subplot(4, length(m), length(m) + i)
    plot(t, abs(X - X_real))
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

    subplot(4, length(m), 2 * length(m) + i)
    plot(fs(1:d), P_lhs(1:d)) 
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

    subplot(4, length(m), 3 * length(m) + i)
    plot(fs(1:d), abs(P_lhs(1:d) - P_rhs(1:d))) 
    set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)

end

print(agrann.figure_file('ex1_case1_Growing.eps'), '-depsc2')

for i = 1 : length(m)
    fprintf('%.2f & ', res(1, i))
end
fprintf('\n');
for i = 1 : length(m)
    fprintf('%.4e & ', res(2, i))
end
fprintf('\n');

toc

end
