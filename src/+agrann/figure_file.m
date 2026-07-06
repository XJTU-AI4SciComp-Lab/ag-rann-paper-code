function path = figure_file(filename)
%FIGURE_FILE Return a writable path inside figures/generated/.

directory = fullfile(agrann.project_root(), "figures", "generated");
if ~isfolder(directory)
    mkdir(directory);
end
path = fullfile(directory, filename);
end
