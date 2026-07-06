clear
clc
% close all

N = [2 5 3 3 1];
gap = 4;
t = 3;
X = 0:gap:gap*length(N)-1;

gg = [0 100/255 0];
bb = [0 0 205/255];
rr = [1 0 0];

figure('Units','centimeter','Position',[10 10 27 15])
hold on

i = 1;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 1)) / 2 + k], 'color', 'k', 'LineWidth', 1.5)
    end
end

scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + (1 : N(i)), 300, 'k', 'LineWidth', 2, 'MarkerFaceColor', 'w')

%% blue
i = 2;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 1)) / 2 + k + t], 'color', bb, 'LineWidth', 1.5)
    end
end

i = 3;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j + t (max(N) - N(i + 1)) / 2 + k - t], 'color', gg, 'LineWidth', 1.5)
    end
end

for j = 1 : N(i - 1)
    for k = 1 : N(i + 1)
        plot([X(i - 1) X(i + 1)], [(max(N) - N(i - 1)) / 2 + j (max(N) - N(i + 1)) / 2 + k - t], 'color', gg, 'LineWidth', 1.5)
    end
end

scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + t + (1 : N(i)), 300, bb, 'LineWidth', 2, 'MarkerFaceColor', 'w')

i = 4;
scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 - t + (1 : N(i)), 300, gg, 'LineWidth', 2, 'MarkerFaceColor', 'w')

%% yellow
i = 2;
% for j = 1 : N(i)
%     for k = 1 : N(i + 2)
%         plot([X(i) X(i + 2)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 2)) / 2 + k], 'color', rr, 'LineWidth', 1)
%     end
% end

scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + 1 : (max(N) - N(i)) / 2 + N(i), 300, 'k', 'LineWidth', 2, 'MarkerFaceColor', 'w')

%% line
d4 = 0.7;
d1 = 0;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + [1-d4 1-d4 N(i) + d4 N(i) + d4 1-d4], 'Color', rr, 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [d4 3 * gap 3 * gap], (max(N) + 1) / 2 + [0 0 -d1], 'color', rr, 'LineWidth', 2, 'LineStyle', ':')
i = 3;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + t + [1-d4 1-d4 N(i) + d4 N(i) + d4 1-d4], 'color', bb, 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [d4 2*gap 2*gap], max(N) / 2 + t + 0.5 + [0 0 -(t + d1)], 'color', bb, 'LineWidth', 2, LineStyle=':')
i = 4;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 - t + [1-d4 1-d4 N(i) + d4 N(i) + d4 1-d4], 'color', gg, 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [d4 gap gap], max(N) / 2 - t + 0.5 + [0 0 (t + d1)], 'color', gg, 'LineWidth', 2, LineStyle=':')

scatter(X(end), (max(N) + 1) / 2 - d1, 300, 'k', 'LineWidth', 1.5, 'MarkerFaceColor', 'w')

% text

text(3.7, 6.1, '$\phi_1(x)$', Interpreter = 'latex', Color = rr, FontSize = 15)
text(8.8, 7.5, '$\phi_2(x)$', Interpreter = 'latex', Color = bb, FontSize = 15)
text(12.8, -1.3, '$\phi_3(x)$', Interpreter = 'latex', Color = gg, FontSize = 15)
text(1.5, 4.5, '$W^{(0,1)}$', Interpreter = 'latex', Color = [0 0 0], FontSize = 15, FontWeight='bold')
text(3.5, 5.35, '$b^{(1)}$', Interpreter = 'latex', Color = [0 0 0], FontSize = 15, FontWeight='bold')
text(5.5, 6.3, '$W^{(1,2)}$', Interpreter = 'latex', Color = bb, FontSize = 15, FontWeight='bold')
text(7.5, 7.35, '$b^{(2)}$', Interpreter = 'latex', Color = bb, FontSize = 15, FontWeight='bold')
text(6.6, -0.2, '$W^{(1,3)}$', Interpreter = 'latex', Color = gg, FontSize = 15, FontWeight='bold')
text(9.7, 4.7, '$W^{(2,3)}$', Interpreter = 'latex', Color = gg, FontSize = 15, FontWeight='bold')
text(12.1, 1.4, '$b^{(3)}$', Interpreter = 'latex', Color = gg, FontSize = 15, FontWeight='bold')
text(14, 3.3, '$W^{(1,4)}$', Interpreter = 'latex', Color = rr, FontSize = 15, FontWeight='bold')
text(14, 6.3, '$W^{(2,4)}$', Interpreter = 'latex', Color = bb, FontSize = 15, FontWeight='bold')
text(14, 0.3, '$W^{(3,4)}$', Interpreter = 'latex', Color = gg, FontSize = 15, FontWeight='bold')


% hold off
axis off
axis equal