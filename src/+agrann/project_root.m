function root = project_root()
%PROJECT_ROOT Return the absolute repository root.

root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
end
