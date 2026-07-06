function reproduce_table_06()
%REPRODUCE_TABLE_06 Run the Example 1 Case 4 parameter sweep.

project_root = fileparts(fileparts(mfilename("fullpath")));
addpath(project_root, "-begin");
startup();
directory = fullfile(project_root, "experiments", "example1_poisson", "case4");
addpath(directory, "-begin");
cleanup = onCleanup(@() rmpath(directory)); %#ok<NASGU>
run(fullfile(directory, "case4.m"));
end
