function startup()
%STARTUP Add the AG-RaNN core package to the MATLAB path.

root = fileparts(mfilename("fullpath"));
addpath(fullfile(root, "src"), "-begin");
addpath(fullfile(root, "experiments"), "-begin");
end
