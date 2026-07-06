%% RATE OF CONVERGENCE OF AFEM: P1 Linear Element for Poisson 
%
% This example is to show that: adaptive finite element methods can
% recovery the optimal convergent rate for elliptic equations even the
% solution is singular.
%
% # Lshape problem.
% # Kellogg problem.

% help Poissonafemrate;
close all;
clear variables

[node,elem] = squaremesh([0 1 0 1], 0.25);
bdFlag = setboundary(node,elem,'Dirichlet');
mesh = struct('node',node,'elem',elem,'bdFlag',bdFlag);
pde = A120data;
option.L0 = 1;
option.maxIt = 100;
option.maxN = 1e7;
option.theta = 0.4;
option.plotflag = 0;
option.rateshift = 20;
option.viewangle = [27,26];

tic
err = afemPoisson(mesh,pde,option);
toc
figure(3);
showrate2(err.N,err.H1,20,'k-*','||Du-Du_h||',err.N,err.eta,20,'-+','eta');
% latexerrtable(err.N,[err.H1 err.eta])