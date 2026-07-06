% close all

N = [2 3 1];
gap = 2;
t = -3;
X = 0:gap:gap*length(N)-1;

gg = [0 100/255 0];
bb = [0 0 205/255];
rr = [1 0 0];
pp = [139/255 0 139/255];

figure('Units','centimeter','Position',[10 10 33 10])
hold on

i = 1;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 1)) / 2 + k],  Color='k', LineWidth=1.5)
    end
end
scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + 1 : (max(N) - N(i)) / 2 + N(i), 300, 'k', LineWidth=2, MarkerFaceColor='w')

i = 2;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 1)) / 2 + k], 'color', [0.9290 0.6940 0.1250], LineWidth=2, LineStyle = ':')
    end
end

scatter(X(i) * ones(1, N(i)-2), (max(N) - N(i)) / 2 + 1 : (max(N) - N(i)) / 2 + N(i), 300, 'k', LineWidth=2, MarkerFaceColor='w')
scatter(X(end) * ones(1, N(end)), (max(N) - N(end)) / 2 + 1 : (max(N) - N(end)) / 2 + N(end), 300, 'k', LineWidth=2, MarkerFaceColor='w')

%%
N = [2 5 1];
X = X + gap * length(N) + 1;

i = 1;
for j = 1 : N(i)
    for k = 1 : 2
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j - 1 (max(N) - N(i + 1)) / 2 + k - 1], Color=pp, LineWidth=1.5)
    end
    for k = 3 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j - 1 (max(N) - N(i + 1)) / 2 + k - 1],  Color='k', LineWidth=1.5)
    end
end
scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 : (max(N) - N(i)) / 2 + N(i) - 1, 300, 'k', LineWidth=2, MarkerFaceColor='w')

i = 2;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j - 1 (max(N) - N(i + 1)) / 2 + k - 1], 'color', [0.9290 0.6940 0.1250], LineWidth=2, LineStyle = ':')
    end
end

scatter(X(i) * ones(1, N(i)-2), (max(N) - N(i)) / 2 + 2 : (max(N) - N(i)) / 2 + N(i) - 1, 300, 'k', LineWidth=2, MarkerFaceColor='w')
scatter(X(i) * ones(1, 2), (max(N) - N(i)) / 2 : (max(N) - N(i)) / 2 + 1, 300, Color=pp, LineWidth=2, MarkerFaceColor='w')
scatter(X(end) * ones(1, N(end)), (max(N) - N(end)) / 2 : (max(N) - N(end)) / 2 + N(end) - 1, 300, 'k', LineWidth=2, MarkerFaceColor='w')

d4 = 0.37;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + [2-d4 2-d4 4+d4 4+d4 2-d4], 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + [-d4 -d4 1+d4 1+d4 -d4], 'Color', pp, 'LineWidth', 1.5, 'LineStyle', '--')

text(8, 4.7, 'existing neurons', Color = 'k', FontSize = 11)
text(8.2, -0.7, 'new neurons', Color = pp, FontSize = 11)
text(7.6, 3.5, '$W$', Interpreter = 'latex', Color = 'k', FontSize = 15)
text(9.4, 4, '$b$', Interpreter = 'latex', Color = 'k', FontSize = 15)
text(7.6, 0.3, '$W_{add}$', Interpreter = 'latex', Color = pp, FontSize = 15)
text(9.4, 0, '$b_{add}$', Interpreter = 'latex', Color = pp, FontSize = 15)

%%
N = [2 5 3 1];
X = 0:gap:gap*length(N)-1;
X = X + 2 * gap * (length(N) - 1) + 2;

i = 1;
for j = 1 : N(i)
    for k = 1 : 2
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j - 1 (max(N) - N(i + 1)) / 2 + k], 'color', pp, LineWidth=1.5)
    end
    for k = 3 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j - 1 (max(N) - N(i + 1)) / 2 + k], 'color', 'k', LineWidth=1.5)
    end
end
scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 : (max(N) - N(i)) / 2 + N(i) - 1, 300, 'k', LineWidth=2, MarkerFaceColor='w')

i = 2;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 1)) / 2 + k + t], Color=bb, LineWidth=1.5)
    end
end

i = length(N) - 1;
for j = 1 : N(i)
    for k = 1 : N(i + 1)
        plot([X(i) X(i + 1)], [(max(N) - N(i)) / 2 + j + t (max(N) - N(i + 1)) / 2 + k - 1], Color=rr, LineWidth=2, LineStyle=':')
    end
end

scatter(X(i) * ones(1, N(i)), (max(N) - N(i)) / 2 + 1 + t : (max(N) - N(i)) / 2 + N(i) + t, 300, bb, LineWidth=2, MarkerFaceColor='w')

i = 2;
for j = 1 : N(i)
    for k = 1 : N(i + 2)
        plot([X(i) X(i + 2)], [(max(N) - N(i)) / 2 + j (max(N) - N(i + 2)) / 2 + k - 1], 'color', [0.9290 0.6940 0.1250], LineWidth=2, LineStyle = ':')
    end
end

scatter(X(i) * ones(1, 2), (max(N) - N(i)) / 2 + 1 : (max(N) - N(i)) / 2 + 2, 300, pp, LineWidth=2, MarkerFaceColor='w')
scatter(X(i) * ones(1, N(i)-2), (max(N) - N(i)) / 2 + 3 : (max(N) - N(i)) / 2 + N(i), 300, 'k', LineWidth=2, MarkerFaceColor='w')
scatter(X(end) * ones(1, N(end)), (max(N) - N(end)) / 2 : (max(N) - N(end)) / 2 + N(end) - 1, 300, 'k', LineWidth=2, MarkerFaceColor='w')

d4 = 0.37;
plot(X(i) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + [1-d4 1-d4 5+d4 5+d4 1-d4], 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--')
plot(X(i+1) + [-d4 d4 d4 -d4 -d4], (max(N) - N(i)) / 2 + [-1-d4 -1-d4 1+d4 1+d4 -1-d4], 'Color', bb, 'LineWidth', 1.5, 'LineStyle', '--')

text(14, 0.3, 'existing layer', Color = 'k', FontSize = 11)
text(16.2, -0.8, 'new layer', Color = bb, FontSize = 11)

% hold off
axis off
axis equal

%%
quiver(5, 2, 1, 0, 0, 'LineWidth', 2.5, 'Color', [0.8 0.2 0.2], 'MaxHeadSize', 1.5, 'AutoScale', 'off');
quiver(12, 2, 1, 0, 0, 'LineWidth', 2.5, 'Color', [0.8 0.2 0.2], 'MaxHeadSize', 1.5, 'AutoScale', 'off');

text(4.6, 2.4, 'neuron growth', Color = 'k', FontSize = 10)
text(11.7, 2.4, 'layer growth', Color = 'k', FontSize = 10)
