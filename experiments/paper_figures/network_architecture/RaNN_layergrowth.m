% close all

N = [2 5 3 1];
gap = 5;
t = -2.5;
X = 0:gap:gap*length(N)-1;

gg = [0 100/255 0];
bb = [0 0 205/255];
rr = [1 0 0];
pp = [139/255 0 139/255];

figure('Units','centimeter','Position',[10 10 27 15])
hold on

d1 = 3;
i = 1;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j - d1 (max(N) - N(i + 1)) / 2 + k], 'color', 'k', 'LineWidth', 1.5)
    end
end

%% magenta
d2 = 7;
for j = 1 : N(i)
    for k = 1 : N(i + 2)
        plot([X(i) X(i + 2)], [(max(N) - N(i)) / 2 + j - d1 (max(N) - N(i + 2)) / 2 + k - d2], 'color', pp, 'LineWidth', 1.5)
    end
end
scatter(X(i + 2) * ones(1, N(i + 2)), (max(N) - N(i + 2)) / 2 + (1 : N(i + 2)) - d2, 300, pp, 'LineWidth', 2, 'MarkerFaceColor', 'w')
scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 - d1 + (1 : N(i)), 300, 'k', 'LineWidth', 2, 'MarkerFaceColor', 'w')

%% blue
i = 2;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 1)) / 2 + k + t], 'color', bb, 'LineWidth', 1.5)
    end
end

i = 3;
% for j = 1 : N(i)
%     for k = 1 : N(i + 1)
%         plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j + t (max(N) - N(i + 1)) / 2 + k], 'color', gg, 'LineWidth', 1)
%     end
% end

scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + t + (1 : N(i)), 300, bb, 'LineWidth', 2, 'MarkerFaceColor', 'w')

%% yellow
i = 2;
% for j = 1 : N(i)
%     for k = 1 : N(i + 2)
%         plot([X(i) X(i + 2)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 2)) / 2 + k], 'color', rr, 'LineWidth', 1)
%     end
% end

scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + 1 : (max(N) - N(i)) / 2 + N(i), 300, 'k', 'LineWidth', 2, 'MarkerFaceColor', 'w')

%% line
d4 = 0.8;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + [1-d4 1-d4 N(i) + d4 N(i) + d4 1-d4], 'Color', rr, 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [d4 3 * gap 3 * gap], (max(N) + 1) / 2 + [0 0 -d1], 'color', rr, 'LineWidth', 2, 'LineStyle', ':')
i = 3;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + t + [1-d4 1-d4 N(i) + d4 N(i) + d4 1-d4], 'color', bb, 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 - d2 + [1-d4 1-d4 N(i) + d4 N(i) + d4 1-d4], 'color', pp, 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [d4 gap gap], max(N) / 2 + t + 0.5 + [0 0 (- t - d2) / 2], 'color', bb, 'LineWidth', 1.5)
plot(X(i) + [d4 gap gap], max(N) / 2 - d2 + 0.5 + [0 0 (t + d2) / 2], 'color', pp, 'LineWidth', 1.5)
plot(X(i) + gap + [0 gap gap], (max(N) + 1) / 2 + [(t - d2) / 2, (t - d2) / 2, -d1], 'color', gg, 'LineWidth', 2, 'LineStyle', ':')
scatter(X(i + 1), (max(N) + 1) / 2 + (t - d2) / 2, 300, gg, 'LineWidth', 2, 'MarkerFaceColor', 'w')
d3 = 0.2;
text(X(i + 1) - 0.36, max(N) / 2 + 0.5 + (t - d2) / 2 + 0.12, '\times', 'Color', gg,'FontSize', 25, FontWeight='bold')
scatter(X(end) + gap, (max(N) + 1) / 2 - d1, 300, 'k', 'LineWidth', 1.5, 'MarkerFaceColor', 'w')

%% text
text(6, 5, '$\phi(x)$', Interpreter = 'latex', Color = rr, FontSize = 15)
text(11, 1, '$\tilde{\psi}(x)$', Interpreter = 'latex', Color = bb, FontSize = 15)
text(11, -4.5, '$G(x)$', Interpreter = 'latex', Color = pp, FontSize = 15)
text(13.6, -1.7, '$\psi(x)$', Interpreter = 'latex', Color = gg, FontSize = 15)
text(2, 3.2, '$W^{(0)}$', Interpreter = 'latex', Color = [0 0 0], FontSize = 15, FontWeight='bold')
text(4.33, 5.4, '$b^{(0)}$', Interpreter = 'latex', Color = [0 0 0], FontSize = 15, FontWeight='bold')
text(7.5, 3.5, '$W^{(1)}$', Interpreter = 'latex', Color = bb, FontSize = 15, FontWeight='bold')
text(10.1, 1.95, '$b^{(1)}$', Interpreter = 'latex', Color = bb, FontSize = 15, FontWeight='bold')
text(18, 3.4, '$w^{(0)}$', Interpreter = 'latex', Color = rr, FontSize = 15, FontWeight='bold')
text(18, -1.4, '$w^{(1)}$', Interpreter = 'latex', Color = gg, FontSize = 15, FontWeight='bold')
text(8.6, -2.34, '$h_1$', Interpreter = 'latex', Color = pp, FontSize = 15, FontWeight='bold')
text(8.6, -3.22, '$h_2$', Interpreter = 'latex', Color = pp, FontSize = 15, FontWeight='bold')
text(8.6, -4.1, '$h_3$', Interpreter = 'latex', Color = pp, FontSize = 15, FontWeight='bold')

% hold off
axis off
axis equal