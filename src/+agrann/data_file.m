function path = data_file(varargin)
%DATA_FILE Return an absolute path inside data/.

path = fullfile(agrann.project_root(), "data", varargin{:});
end
