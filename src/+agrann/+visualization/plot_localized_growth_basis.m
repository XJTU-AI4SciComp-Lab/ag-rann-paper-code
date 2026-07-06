function plot_localized_growth_basis(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2)
%PLOT_LOCALIZED_GROWTH_BASIS Plot the deterministic 6-by-6 basis panel.

V = atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv2(atv1([X(:) Y(:)] * W + b) * Ur * G - V .* G);
figure
tiledlayout(6, 6);
for i = 1 : 36
    nexttile
    surf(X, Y, reshape(V(:, i), size(X, 1), size(X, 2)), EdgeColor = 'none');
    view(2)
end
cb = colorbar;
cb.Layout.Tile = 'east';
end
