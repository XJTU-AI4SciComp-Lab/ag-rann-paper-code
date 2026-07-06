clear
clc
close all

N = 100;
% z = zeros(20, N);
% V = zeros(512*201, N);
% 
% for i = 1 : N
%     [z(:, i), V(:, i)] = AllenCahn(i);
% end
% 
% save result z V

load(agrann.result_file('example4_allen_cahn', 'result.mat'))
load(agrann.data_file('allen_cahn', 'AC_cosine.mat'))
[T, X] = meshgrid(tt, x);

V = sum(V, 2) / N;

% gcf = figure('Units', 'centimeter', 'Position', [5 5 45 10]);
% set(gcf, 'renderer', 'painters')
% 
% subplot(1, 3, 1)
% Draw2D(T, X, uu);
% set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
% title('reference solution')
% subplot(1, 3, 2)
% Draw2D(T, X, V);
% set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
% title('numerical solution')
% subplot(1, 3, 3)
% Draw2D(T, X, abs(V - uu(:)));
% set(gca,'FontSize',10,'FontWeight','bold','linewidth',1);
% title('absolute error')
% 
% print('ex4_solution.eps', '-depsc2')

err_L2_rel = sqrt(sum((V - uu(:)) .^ 2) / sum(uu(:) .^ 2));

% gcf = figure('Units', 'centimeter', 'Position', [5 5 30 15]);
% set(gcf, 'renderer', 'painters')
% hold on
% plot(1:100, z(20, :),'Marker','o','Color',[0 0 1],'LineWidth',1);
% set(gca,'yscale','log', 'FontSize',10,'FontWeight','bold','linewidth',1);
% xlabel('random seed','FontSize',12,'fontname','Times','FontWeight','bold')
% ylabel('relative L^{2} errors','FontSize',12,'fontname','Times','FontWeight','bold')
% % plot([0 100], [err_L2_rel, err_L2_rel], '--','Color',[0 0 0]);
% p = plot([0 100], [sum(z(20, :)) / N, sum(z(20, :)) / N], '--','Color',[1 0 0]);
% legend(p, 'average','Location','northeast');
% 
% print('ex4_average.eps', '-depsc2')
