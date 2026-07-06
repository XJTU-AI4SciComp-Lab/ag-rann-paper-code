function reproduce_table_03()
%REPRODUCE_TABLE_03 Run the Example 1 Case 2 parameter sweep.

project_root = fileparts(fileparts(mfilename("fullpath")));
addpath(project_root, "-begin");
startup();
directory = fullfile(project_root, "experiments", "example1_poisson", "case2");
addpath(directory, "-begin");
cleanup = onCleanup(@() rmpath(directory)); %#ok<NASGU>
run(fullfile(directory, "case2.m"));
end
