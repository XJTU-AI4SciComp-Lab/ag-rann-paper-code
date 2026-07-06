function draw_2d(X, Y, V)
%DRAW_2D Draw the shared top-view surface used by Poisson experiments.

% figure
% hold on
surf(X, Y, reshape(V, size(X, 1), size(X, 2)), EdgeColor = 'none');
% view(-38, 30)
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
view(2)
colorbar
axis square
end
