% Minimal deterministic AG-RaNN-style Poisson example.

example_directory = fileparts(mfilename("fullpath"));
project_root = fileparts(fileparts(example_directory));
addpath(fullfile(project_root, "src"), "-begin");
addpath(example_directory, "-begin");

parameters = config();
rng(parameters.seed);

exact_solution = @(x) sin(pi * x);
forcing = @(x) pi ^ 2 * sin(pi * x);
activation = @(x) tanh(x);
activation_second = @(x) -2 * tanh(x) .* (1 - tanh(x) .^ 2);

interior = linspace(0, 1, parameters.num_collocation_points)';
boundary = [0; 1];
weights = unifrnd(-parameters.random_range, parameters.random_range, ...
    1, parameters.num_features);
centers = unifrnd(0, 1, 1, parameters.num_features);
biases = -centers .* weights;

interior_matrix = -(weights .^ 2) .* activation_second(interior * weights + biases);
boundary_matrix = activation(boundary * weights + biases);
system_matrix = [interior_matrix; parameters.boundary_penalty * boundary_matrix];
right_hand_side = [forcing(interior); ...
    parameters.boundary_penalty * exact_solution(boundary)];

coefficients = agrann.solve_output_layer(system_matrix, right_hand_side);

evaluation_points = linspace(0, 1, parameters.num_evaluation_points)';
prediction = activation(evaluation_points * weights + biases) * coefficients;
reference = exact_solution(evaluation_points);
relative_l2_error = norm(prediction - reference) / norm(reference);

metrics = table("relative_l2_error", relative_l2_error, ...
    'VariableNames', {'metric', 'value'});
writetable(metrics, agrann.result_file("minimal_poisson_1d", "metrics.csv"));
fprintf("minimal_poisson_1d relative_L2_error = %.16e\n", relative_l2_error);
