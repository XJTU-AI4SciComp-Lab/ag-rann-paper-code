function AdvectionReaction

if nargin < 1
    num_inter = 50; 
    num_bound = 200;
    num_neuron = 500;
    r_neuron = 10;
    dimension = 2;
    penalty_bound_init = 1;
    num_neuron_add = 500;
    lambda = 0;
    scale1 = 1;
    scale2 = 1;
    grad_random = 5;
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
m_add = num_neuron_add;
r = r_neuron;
d = dimension;
eta0 = penalty_bound_init;

atv1 = @(x)tanh(x);
diff1_atv1 = @(x) 1 - tanh(x) .^ 2;

[atv0, diff1_atv0, ~] = agrann.wavelet_basis(0);
[atv2, diff1_atv2, ~] = agrann.wavelet_basis(0);

[err_pt, err_wt] = agrann.gauss_legendre_2d(100, left, right, bottom, top);

W = unifrnd(-r, r, d, m);
b = [unifrnd(left, right, 1, m); unifrnd(bottom, top, 1, m)];
b = - sum(b .* W, 1);

eps = 1e-10;
ptt1 = linspace(left + eps, right - eps, (right - left) / (top - bottom) * ni);
ptt2 = linspace(bottom + eps, top - eps, ni);
[pttx, ptty] = meshgrid(ptt1, ptt2);
pti = [pttx(:) ptty(:)];
ptt1 = linspace(left, right, nb)';
ptb = [ptt1 bottom*ones(nb, 1)];
ni = size(pti, 1);

U_err = solu(err_pt);
V_err1 = atv1(err_pt * W + b);

Ai = W(2, :) .* diff1_atv1(pti * W + b);
Fi = rhs(pti);

Ab = atv1(ptb * W + b);
Fb = solu(ptb);

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

if lambda ~= 0
    F = A.' * F;
    A = A.' * A + lambda * eye(size(A, 2));
end

Ur = agrann.solve_output_layer(A, F, GPU, QR);

err_L2_rel = sqrt(err_wt' * (V_err1 * Ur - U_err) .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);    

[X, Y] = meshgrid(linspace(left, right, 500), linspace(bottom, top, 500));

subplot(2, 3, 1)
Draw2D(X, Y, solu([X(:) Y(:)]));

V = atv1([X(:) Y(:)] * W + b) * Ur;

subplot(2, 3, 2)
Draw2D(X, Y, V);
subplot(2, 3, 3)
Draw2D(X, Y, abs(V - solu([X(:) Y(:)])));

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
subplot(2, 3, 4)
scatter(pti(I, 1), pti(I, 2), 20, 'red')
axis equal
axis([left, right, bottom, top])

if grad_random == 0
    grad = scale1 * [repmat(W(1, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur ...
            repmat(W(2, :), m2, 1) .* diff1_atv1(pti(I, :) * W + b) * Ur]';
else
    grad = unifrnd(-grad_random, grad_random, d, m2);
end

G = sqrt(sum(grad .^ 2, 1));
V = (atv1(pti(I, :) * W + b) * Ur)';

grad = scale2 * grad;

V_atv0_1 = atv0(grad(1, :) .* (pti(:, 1) - pti(I, 1)'));
V_atv0_2 = atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)'));
V_atv2_0 = atv1(pti * W + b) * Ur * G - V .* G;
V_atv2 = atv2(V_atv2_0);
V_diff1_atv2_0 = diff1_atv2(V_atv2_0);

V_err2 = atv2(V_err1 * Ur * G - V .* G);

Ai2 = pti(:, 1) .* V_atv0_1 .* grad(2, :) .* diff1_atv0(grad(2, :) .* (pti(:, 2) - pti(I, 2)')) .* V_atv2 + ...
      V_atv0_1 .* V_atv0_2 .* V_diff1_atv2_0 .* G .* (Ai * Ur);

Ab2 = atv0(grad(1, :) .* (ptb(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (ptb(:, 2) - pti(I, 2)')) .* atv2(atv1(ptb * W + b) * Ur * G - V .* G);

Ai(:, I2) = Ai2;
Ab(:, I2) = Ab2;

A = [Ai; eta0 * Ab];
F = [Fi; eta0 * Fb];

Ur_old = Ur;

Ur = agrann.solve_output_layer(A, F, GPU, QR);

err_L2_rel = sqrt(err_wt' * (V_err1(:, I1) * Ur(I1) + atv0(grad(1, :) .* (err_pt(:, 1) - pti(I, 1)')) .* atv0(grad(2, :) .* (err_pt(:, 2) - pti(I, 2)')) .* V_err2 * Ur(I2) - U_err) .^ 2 / (err_wt' * U_err .^ 2));
fprintf('L2_relative_error = %e\n', err_L2_rel);

V = atv1([X(:) Y(:)] * W(:, I1) + b(:, I1)) * Ur(I1) + ...
    atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)]  * W + b) * Ur_old * G - V .* G) * Ur(I2);

subplot(2, 3, 5)
Draw2D(X, Y, V)
subplot(2, 3, 6)
Draw2D(X, Y, abs(V - solu([X(:) Y(:)])))

toc

end
