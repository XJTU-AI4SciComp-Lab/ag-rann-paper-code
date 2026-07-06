result = zeros(9, 8);

M = 500;

for i = 1 : 3
    for j = 1 : 3
        result(3 * (i - 1) + j, [2 5 3]) = Poisson2D_case2(0, 0, 10, M * (i + 1), M * j);
        result(3 * (i - 1) + j, [2 6 3]) = Poisson2D_case2(1, 1, 10, M * (i + 1), M * j);
        result(3 * (i - 1) + j, [2 7 3]) = Poisson2D_case2(1, 0, 0, M * (i + 1), M * j);
        result(3 * (i - 1) + j, [2 8 3]) = Poisson2D_case2(1, 0, 10, M * (i + 1), M * j);
        result(3 * (i - 1) + j, [1 4]) = [M * (i + 1), M * j];
    end
end

save(agrann.result_file('example1_poisson_case2', 'case2.mat'), 'result')

for i = 1 : 3
    for j = 1 : 3
        fprintf('%d & %.4e & %d & %d & %.4e & %.4e & %.4e & %.4e \\\\hline\n', result(3 * (i - 1) + j, 1), result(3 * (i - 1) + j, 2), result(3 * (i - 1) + j, 3), result(3 * (i - 1) + j, 4), result(3 * (i - 1) + j, 5), result(3 * (i - 1) + j, 6), result(3 * (i - 1) + j, 7), result(3 * (i - 1) + j, 8))
    end
end
