function GrowingBasis1(grad, X, Y, pti, I, W, b, Ur, V, G, atv0, atv1, atv2)

V1 = atv1([X(:) Y(:)] * W + b) * Ur * G - V .* G;
figure('Units','centimeter','Position',[10 10 23 5])
tiledlayout(2, 8);
for i = 17 : 32
    nexttile
    surf(X, Y, reshape(V1(:, i), size(X, 1), size(X, 2)), EdgeColor = 'none');
    set(gca,'FontSize',7,'FontWeight','bold','linewidth',1)
    view(2)
    axis square
end
