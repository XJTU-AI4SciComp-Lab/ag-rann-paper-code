function Draw3D(X, Y, Z, V)

slice(X, Y, Z, reshape(V, size(X, 1), size(X, 1), size(X, 1)), 0.5, 0.5, 0.5)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
shading flat
colorbar
view(-38, 30)
axis square