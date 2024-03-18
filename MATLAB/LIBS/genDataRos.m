%------------------------------- genDataRos ------------------------------%
%
% Generate compact representation data for eigenvalue computations
% The solver stores the data in DATA/
%
% This file has to be run from within the LIBS folder
%
%-------------------------------------------------------------------------%
% 02/05/24, J.B., Initial version
% 02/11/24, J.B., Update to use the compact representation

%clc;
clear;

addpath(genpath('../SOLVER/'));
addpath(genpath('../LIBS/'));

%
% Initial values
%
opts.print  = 1;
opts.maxIt  = 1000;
opts.tol    = 1e-3;
opts.whichL = 'wolfeG'; % wolfeG, wolfeB, mt 
opts.storeCOMP= false;
opts.whichV   = 's';

ns  = 2.^(3:13); % Problem dimensions

%
% Loop through problem dimensions
%
for i=1:length(ns)

    % Problem setup
    n       = ns(i);   
    func    = @(x)(sum(100*(x(1:2:n).^2-x(2:2:n)).^2+(1-x(1:2:n)).^2));
    grad    = @(x)(rosen_ipopt_grad(x));    
    x       = zeros(n,1);
    
    % Call the solver
    [x,fx,outs] = compLS(x,func,grad,opts);

end