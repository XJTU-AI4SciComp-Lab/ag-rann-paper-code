function Poisson2D

if nargin < 1
    num_inter = 300; 
    num_bound = 3000;
    num_neuron = 1000;
    r_neuron = 15;
    dimension = 2;
    penalty_bound_init = 3000;
    num_neuron_add = 1500;
    lambda = 0;
    S = 120;
end    

rng(1)

solu = @(x)agrann.problems.poisson_sharp_solution(x, S);
rhs = @(x)agrann.problems.poisson_sharp_forcing(x, S);
left = 0;
right = 1;


GPU = 0;
QR = 1;

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
diff2_atv1 = @(x) - 2 * tanh(x) .* (1 - tanh(x) .^ 2);

W = unifrnd(-r, r, d, m);
b = unifrnd(left, right, d, m);
b = - sum(b .* W, 1);
W(:, 1) = 0;
b(:, 1) = 0;

[pti, ptb] = agrann.uniform_points(2, left, right, ni, nb);
ni = size(pti, 1);

V_diff2_atv1 = sum(W .^ 2, 1) .* diff2_atv1(pti * W + b);

Ai = - V_diff2_atv1;
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

t = linspace(left, right, 200);
[X, Y] = meshgrid(t, t);

Draw2D(X, Y, solu([X(:) Y(:)]));

I2 = find(Ur == 0);
m2 = size(I2, 1);

if m2 < m_add
    m2 = m_add;
end

[~, I] = sort(abs(Ai(1 : ni, :) * Ur - Fi(1 : ni)));
I = I(end - m2 + 1:end);
figure('Units','centimeter','Position',[10 10 13 10])
scatter(pti(I, 1), pti(I, 2), 20, 'red')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis equal
axis([0 1 0 1])


end