function Draw1D_1(X, Y, V, solu)

% figure
hold on
i = (size(Y, 1) + 1) / 2;
V = reshape(V, size(Y, 1), size(Y, 1));
plot(Y(:, i), solu([X(:, i) Y(:, i)]), LineWidth=1.5);
plot(Y(:, i), V(:, i), LineWidth=1.5);
legend('$u$', '$u_\rho$' ,'Interpreter', 'latex')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
ylim([-0.1,1.1])