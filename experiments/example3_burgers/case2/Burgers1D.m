function z = Burgers1D(num_neuron, num_neuron_add, grad_random)

if nargin < 1
    num_neuron = 2000;
    num_neuron_add = 2000;
    grad_random = 7;
end    

num_inter = 200; 
num_bound = 200;
r_neuron = 20;
dimension = 2;
penalty_bound_init = 100;
lambda = 0;
scale1 = 0.1;
scale2 = 1;
E = 0.01;
max_it = [10, 0, 0, 3];

rng(1)

solu = @(x)agrann.problems.burgers_logistic_solution(x, E);
rhs = @(x)agrann.problems.zero_forcing(x);

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
m_add = num_neuron_add;
r = r_neuron;
d = dimension;
eta0 = penalty_bound_init;

it_err = zeros(1, sum(max_it));
num_it_err = 1;

atv1 = @(x)tanh(x);
diff1_atv1 = @(x) 1 - tanh(x) .^ 2;
diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

[atv0, diff1_atv0, diff2_atv0] = agrann.wavelet_basis(0);
[atv2, diff1_atv2, diff2_atv2] = agrann.wavelet_basis(0);

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, left, right);

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
       [ptt1 top*ones(size(ptt1))];
       [zeros(nb, 1) linspace(bottom, top, nb)']];

U_err = solu(err_pt);
V_err1 = atv1(err_pt * W + b);

Fi0 = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

Ur = zeros(m, 1);

V_atv1 = atv1(pti * W + b);
V_diff1_atv1 = diff1_atv1(pti * W + b);
V_diff1_atv1_1 = W(1, :) .* V_diff1_atv1;
V_diff1_atv1_2 = W(2, :) .* V_diff1_atv1;
V_diff2_atv1_2 = W(2, :) .^ 2 .* diff2_atv1(pti * W + b);

for i = 1 : max_it(1)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 - E * V_diff2_atv1_2;

    A = [Ai; eta0 * Ab];
    F = [Fi0; eta0 * Fb];

    if lambda ~= 0
        F = A.' * F;
        A = A.' * A + lambda * eye(size(A, 2));
    end

    Ur = agrann.solve_output_layer(A, F, GPU, QR);

    err_point = V_err1 * Ur - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

for i = 1 : max_it(2)
    Ai = V_diff1_atv1_1 + V_atv1 * Ur .* V_diff1_atv1_2 + V_atv1 .* (V_diff1_atv1_2 * Ur) - E * V_diff2_atv1_2;

    Fi = Fi0 + (V_atv1 * Ur) .* (V_diff1_atv1_2 * Ur);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];
    
    if lambda ~= 0
        F = A.' * F;
        A = A.' * A + lambda * eye(size(A, 2));
    end

    Ur = agrann.solve_output_layer(A, F, GPU, QR);
    
    err_point = V_err1 * Ur - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

z(1) = err_L2_rel;

X = reshape(err_pt(:, 1), 100, 100);
Y = reshape(err_pt(:, 2), 100, 100);

gcf = figure('Units', 'centimeter', 'Position', [5 5 30 15]);
set(gcf, 'renderer', 'painters')

subplot(2, 3, 1)
Draw2D(X, Y, solu([X(:) Y(:)]));
zlim([-0.2 1.2])
view(2)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
V = atv1([X(:) Y(:)] * W + b) * Ur;
title('exact solution')

subplot(2, 3, 2)
Draw2D(X, Y, V);
view(2)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
title('numerical solution')
subplot(2, 3, 3)
Draw2D(X, Y, abs(V - solu([X(:) Y(:)])));
view(2)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
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

ni = size(pti, 1);
[~, I] = sort(abs(A(1 : ni, :) * Ur - F(1 : ni)));
I = I(end - m2 + 1:end);

if grad_random == 0
    grad = scale1 * [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (atv1(pti(I, :) * W + b) * Ur)';

grad = scale2 * grad;

Ab(:, I2) = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb * W + b) * Ur * G - V .* G);

Fi = rhs(pti);

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
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1) - E * V_diff2_atv1_2(:, I1);

    Ai(:, I2) = V_atv0_2 .* (V_diff1_atv0_1 .* V_atv2 + V_atv0_1 .* V_diff1_atv2_1) + ...
          Un .* V_atv0_1 .* (V_diff1_atv0_2 .* V_atv2 + V_atv0_2 .* V_diff1_atv2_2) - ...
            E * (V_atv0_1 .* V_atv0_2 .* V_diff2_atv2_2 + ...
                2 * V_atv0_1 .* V_diff1_atv0_2 .* V_diff1_atv2_2 + ...
                V_atv0_1 .* V_diff2_atv0_2 .* V_atv2);

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F, GPU, QR);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));

    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

for i = 1 : max_it(4)
    Un = V_atv1(:, I1) * Ur_new(I1) + V_atv0_1 .* V_atv0_2 .* V_atv2 * Ur_new(I2);
    Unx = V_diff1_atv1_2(:, I1) * Ur_new(I1) + V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 * Ur_new(I2) + ...
                                               V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2 * Ur_new(I2);
    Ai(:, I1) = V_diff1_atv1_1(:, I1) + Un .* V_diff1_atv1_2(:, I1) + V_atv1(:, I1) .* Unx - E * V_diff2_atv1_2(:, I1);

    Ai(:, I2) = V_diff1_atv0_1 .* V_atv0_2 .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_1 + ...
          Un .* (V_atv0_1 .* (V_diff1_atv0_2) .* V_atv2 + V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_2) + ...
          V_atv0_1 .* V_atv0_2 .* V_atv2 .* Unx - ...
            E * (V_atv0_1 .* V_atv0_2 .* V_diff2_atv2_2 + ...
                2 * V_atv0_1 .* V_diff1_atv0_2 .* V_diff1_atv2_2 + ...
                V_atv0_1 .* V_diff2_atv0_2 .* V_atv2);

    Fi = Fi0 + Un .* Unx;

    A = [Ai; eta0 * Ab];
    F = [Fi; eta0 * Fb];

    Ur_new = agrann.solve_output_layer(A, F, GPU, QR);

    err_point = V_err1(:, I1) * Ur_new(I1) + V_err0_1 .* V_err0_2 .* V_err2 * Ur_new(I2) - U_err;
    err_L2_rel = sqrt(err_wt' * err_point .^ 2 / (err_wt' * U_err .^ 2));
    err_Linf_rel = max(abs(err_point)) / max(abs(U_err));
    
    fprintf('L2_relative_error = %e, ', err_L2_rel);    
    fprintf('Linf_relative_error = %e\n', err_Linf_rel);
    it_err(num_it_err) = err_L2_rel;
    num_it_err = num_it_err + 1;
end

z(2) = err_L2_rel;

V = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur_new(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)]  * W + b) * Ur * G - V .* G) * Ur_new(I2);

subplot(2, 3, 5)
Draw2D(X, Y, V);
view(2)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
title('numerical solution')
subplot(2, 3, 6)
Draw2D(X, Y, abs(V - solu([X(:) Y(:)])));
view(2)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
title('absolute error')

subplot(2, 3, 4)
hold on
p1 = plot(10:13, it_err(10:13),'Marker','o','Color',[0 0 1],'LineWidth',1);
p2 = plot(it_err(1:10),'Marker','o','Color',[1 0 0],'LineWidth',1);
set(gca,'yscale','log', 'FontSize',10,'FontWeight','bold','linewidth',1);
xlabel('iteration','FontSize',12,'fontname','Times','FontWeight','bold')
ylabel('relative L^{2} errors','FontSize',12,'fontname','Times','FontWeight','bold')
plot([10 10], [1e-10, 1], '--','Color',[0 0 0])
legend([p2, p1], 'RaNN', 'AG-RaNN','Location','southwest');
axis square

print(agrann.figure_file('ex3_case2.eps'), '-depsc2')

toc

end
