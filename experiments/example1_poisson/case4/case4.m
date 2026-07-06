M = 50;

result = zeros(3, 5);

for i = 1 : 3
    for j = 1 : 3
        result(i, [1 2 + j 2]) = Poisson3D_case4(M * 2 ^ (i - 1), M * 2 ^ (j - 1));
    end
end

save(agrann.result_file('example1_poisson_case4', 'case4.mat'), 'result')

for i = 1 : 3
    fprintf('%.4e & %d & %.4e & %.4e & %.4e \\\\\\hline\n', result(i, 1), result(i, 2), result(i, 3), result(i, 4), result(i, 5))
end
