function reproduce_table_11()
%REPRODUCE_TABLE_11 Run the Example 3 Case 2 parameter sweep.

project_root = fileparts(fileparts(mfilename("fullpath")));
addpath(project_root, "-begin");
startup();
directory = fullfile(project_root, "experiments", "example3_burgers", "case2");
addpath(directory, "-begin");
cleanup = onCleanup(@() rmpath(directory)); %#ok<NASGU>
run(fullfile(directory, "case2.m"));
end
