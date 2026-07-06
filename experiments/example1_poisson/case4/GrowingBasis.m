function GrowingBasis(grad, X, Y, Z, pti, I, W, b, Ur, V, G, atv0, atv1, atv2)

V = atv0(grad(1, :) .* (X(:) - pti(I, 1)')) .* atv0(grad(2, :) .* (Y(:) - pti(I, 2)')) .* atv0(grad(3, :) .* (Z(:) - pti(I, 3)')) .* atv2(atv1([X(:) Y(:) Z(:)] * W + b) * Ur * G - V .* G);
figure
tiledlayout(6, 6);
for i = 1 : 36
    nexttile
    slice(X, Y, Z, reshape(V(:, i), size(X, 1), size(X, 1), size(X, 1)), 0.5, 0.5, 0.5);
    shading flat
    view(3)
end
cb = colorbar;
cb.Layout.Tile = 'east';
