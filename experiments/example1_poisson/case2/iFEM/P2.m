%% RATE OF CONVERGENCE OF QUADRATIC ELEMENT FOR POISSON EQUATION
%
% This example is to show the rate of convergence of quadratic finite
% element approximation of the Poisson equation on the unit square with the
% following boundary conditions:
%
% - Non-empty Dirichlet boundary condition.
% - Pure Neumann boundary condition.
% - Robin boundary condition.
%
% The basis, data structure and numerical test is summarized in <a
% href="matlab:ifem PoissonP2femrate">PoissonP2femrate</a>.
%
% See also Poissonfemrate, Poissonafemrate
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

clear variables

%% Setting
[node,elem] = squaremesh([0,1,0,1],0.5); 
mesh = struct('node',node,'elem',elem);
option.L0 = 1;
option.maxIt = 10;
option.printlevel = 1;
option.plotflag = 1;
option.elemType = 'P2';

%% Non-empty Dirichlet boundary condition.
pde = A120data;
mesh.bdFlag = setboundary(node,elem,'Dirichlet');
femPoisson(mesh,pde,option);

%% Conclusion
%
% The optimal rate of convergence of the H1-norm (2nd order) and L2-norm
% (3rd order) is observed. The order of ||DuI-Duh|| is 2.5 order and
% thus superconvergence exists.
%
% MGCG converges uniformly in all cases.