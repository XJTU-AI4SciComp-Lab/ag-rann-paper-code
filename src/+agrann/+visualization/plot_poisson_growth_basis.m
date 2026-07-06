function plot_poisson_growth_basis(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2)
%PLOT_POISSON_GROWTH_BASIS Plot global and localized Poisson basis functions.

V1 = atv2(atv1([X(:) Y(:)] * W + b) * Ur * G - V .* G);
figure
tiledlayout(6, 10);
for i = 1 : 60
    nexttile
    surf(X, Y, reshape(V1(:, i), size(X, 1), size(X, 2)), EdgeColor = 'none');
    view(2)
end
cb = colorbar;
cb.Layout.Tile = 'east';

V2 = atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)] * W + b) * Ur * G - V .* G);
figure
tiledlayout(6, 10);
for i = 1 : 60
    nexttile
    surf(X, Y, reshape(V2(:, i), size(X, 1), size(X, 2)), EdgeColor = 'none');
    view(2)
end
cb = colorbar;
cb.Layout.Tile = 'east';
end
