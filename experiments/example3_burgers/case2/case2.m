result = zeros(16, 6);

M = 500;

for i = 1 : 4
    for j = 1 : 4
        result(4 * (i - 1) + j, [2 5 3]) = Burgers1D(M * i, M * j, 0);
        result(4 * (i - 1) + j, [2 6 3]) = Burgers1D(M * i, M * j, 7);
        result(4 * (i - 1) + j, [1 4]) = [M * i, M * j];
    end
end

save(agrann.result_file('example3_burgers_case2', 'case2.mat'), 'result')

for i = 1 : 4
    for j = 1 : 4
        fprintf('%d & %.4e & %d & %d & %.4e & %.4e \\\\hline\n', result(4 * (i - 1) + j, 1), result(4 * (i - 1) + j, 2), result(4 * (i - 1) + j, 3), result(4 * (i - 1) + j, 4), result(4 * (i - 1) + j, 5), result(4 * (i - 1) + j, 6))
    end
end
