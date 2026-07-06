function reproduce_table_04_ifem()
%REPRODUCE_TABLE_04_IFEM Run the external iFEM comparison scripts.

required_function = "squaremesh";
if exist(required_function, "file") ~= 2
    error("agrann:reproduce_table_04_ifem:MissingIFEM", ...
        "Table 4 requires a separately installed iFEM toolbox.");
end

project_root = fileparts(fileparts(mfilename("fullpath")));
directory = fullfile(project_root, "experiments", ...
    "example1_poisson", "case2", "iFEM");
addpath(directory, "-begin");
cleanup = onCleanup(@() rmpath(directory)); %#ok<NASGU>

scripts = ["P1.m", "P2.m", "P3.m", "adaptiveP1.m"];
for script = scripts
    command = sprintf("run('%s')", strrep(fullfile(directory, script), "'", "''"));
    evalin("base", command);
end
end
