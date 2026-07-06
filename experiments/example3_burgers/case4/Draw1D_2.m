function Draw1D_2(X, Y, V, solu)

% figure
hold on
plot(Y(:, end), solu([X(:, end) Y(:, end)]), LineWidth=1.5);
plot(Y(:, end), V(end - length(Y(:, end)) + 1 : end), LineWidth=1.5);
legend('$u$', '$u_\rho$' ,'Interpreter', 'latex')
set(gca,'FontSize',10,'FontWeight','bold','linewidth',1)
axis square
ylim([-0.1,1.1])