function varargout = run_experiment(experiment_name, varargin)
%RUN_EXPERIMENT Run one paper experiment without polluting the MATLAB path.
%
% Examples:
%   run_experiment("example1_case2", 0, 0, 2, 10, 10)
%   [errors, solution] = run_experiment("example4_seed", 1)

root = fileparts(mfilename("fullpath"));
startup();

switch lower(string(experiment_name))
    case "example1_case1_best"
        relative_directory = fullfile("example1_poisson", "case1");
        entry_point = "Poisson1D_Best";
    case "example1_case1_growth"
        relative_directory = fullfile("example1_poisson", "case1");
        entry_point = "Poisson1D_Growing";
    case "example1_case2"
        relative_directory = fullfile("example1_poisson", "case2");
        entry_point = "Poisson2D_case2";
    case "example1_case2_figure"
        relative_directory = fullfile("example1_poisson", "case2");
        entry_point = "Poisson2D_case2_plot";
    case "example1_case2_time"
        relative_directory = fullfile("example1_poisson", "case2");
        entry_point = "Poisson2D_case2_time";
    case "example1_case3"
        relative_directory = fullfile("example1_poisson", "case3");
        entry_point = "Poisson2D_case3";
    case "example1_case4"
        relative_directory = fullfile("example1_poisson", "case4");
        entry_point = "Poisson3D_case4";
    case "example1_case5"
        relative_directory = fullfile("example1_poisson", "case5");
        entry_point = "Poisson2D";
    case "example2_single"
        relative_directory = "example2_advection_reaction";
        entry_point = "AdvectionReaction";
    case "example2_domain_split"
        relative_directory = "example2_advection_reaction";
        entry_point = "AdvectionReaction_2net";
    case "example3_case1_best"
        relative_directory = fullfile("example3_burgers", "case1");
        entry_point = "Burgers1D_Growing_best";
    case "example3_case1_growth"
        relative_directory = fullfile("example3_burgers", "case1");
        entry_point = "Burgers1D_Growing_plot";
    case "example3_case2"
        relative_directory = fullfile("example3_burgers", "case2");
        entry_point = "Burgers1D";
    case "example3_case3"
        relative_directory = fullfile("example3_burgers", "case3");
        entry_point = "Burgers1D_case3";
    case "example3_case4_known"
        relative_directory = fullfile("example3_burgers", "case4");
        entry_point = "Burgers1D_case4_known";
    case "example3_case4_all"
        relative_directory = fullfile("example3_burgers", "case4");
        entry_point = "Burgers1D_case4_ALL";
    case "example4_seed"
        relative_directory = "example4_allen_cahn";
        entry_point = "AllenCahn";
    case "example4_ensemble"
        relative_directory = "example4_allen_cahn";
        entry_point = "generate_ensemble";
    case "example4_average"
        relative_directory = "example4_allen_cahn";
        entry_point = "average_ensemble";
    case "paper_layer_growth"
        relative_directory = fullfile("paper_figures", "layer_growth");
        entry_point = "Poisson2D";
    case "paper_layer_growth_residual"
        relative_directory = fullfile("paper_figures", "layer_growth");
        entry_point = "Poisson2D_1";
    otherwise
        error("agrann:run_experiment:UnknownExperiment", ...
            "Unknown experiment '%s'. See docs/reproduction.md.", experiment_name);
end

experiment_directory = fullfile(root, "experiments", relative_directory);
addpath(experiment_directory, "-begin");
path_cleanup = onCleanup(@() remove_experiment_path(experiment_directory)); %#ok<NASGU>

clear_experiment_helpers();
entry = str2func(entry_point);
if nargout == 0
    entry(varargin{:});
else
    [varargout{1:nargout}] = entry(varargin{:});
end
end

function clear_experiment_helpers()
clear u u1 u2 f f2 Draw1D_1 Draw1D_2 Draw2D Draw2D_3 Draw2D_single GrowingBasis
end

function remove_experiment_path(experiment_directory)
rmpath(experiment_directory);
clear_experiment_helpers();
end
