function path = result_file(experiment_name, filename)
%RESULT_FILE Return a writable, experiment-specific path inside results/.

directory = fullfile(agrann.project_root(), "results", experiment_name);
if ~isfolder(directory)
    mkdir(directory);
end
path = fullfile(directory, filename);
end
