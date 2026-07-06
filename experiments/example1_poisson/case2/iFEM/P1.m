%% RATE OF CONVERGENCE OF LINEAR ELEMENT FOR POISSON EQUATION
%
% This example is to show the rate of convergence of linear finite element
% approximation of the Poisson equation on the unit square with the
% following boundary conditions:
%
% - Non-empty Dirichlet boundary condition.
% - Pure Neumann boundary condition.
% - Robin boundary condition.
%
% The basis, data structure and numerical test is summarized in <a
% href="matlab:ifem Poissonfemrate">Poissonfemrate</a>.
%
% See also PoissonP2femrate, Poissonafemrate
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

clear variables
close all

%% Setting
[node,elem] = squaremesh([0,1,0,1],0.5); 
mesh = struct('node',node,'elem',elem);
option.L0 = 1;
option.maxIt = 10;
option.printlevel = 1;

%% Non-empty Dirichlet boundary condition.
option.plotflag = 1;
pde = A120data;
mesh.bdFlag = setboundary(node,elem,'Dirichlet');
femPoisson(mesh,pde,option);

%% Conclusion
%
% The optimal rate of convergence of the H1-norm (1st order) and L2-norm
% (2nd order) is observed. The 2nd order convergent rate between two
% discrete functions ||DuI-Duh|| is known as superconvergence.
%
% MGCG converges uniformly in all cases.
