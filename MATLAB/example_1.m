%--------------------------- example_1 -----------------------------------%
%
% Test to use the compact representation on the Rosenbrock function
% This test enables different choices for the line-search
%
%-------------------------------------------------------------------------%
% 02/05/24, J.B., Initial version
% 02/11/24, J.B., Update to use the compact representation
% 03/18/24, J.B., Preparation for release

clc;
clear;

addpath(genpath('./SOLVER/'));
addpath(genpath('./LIBS/'));


% Problem setup
n = 500;

% Objective and gradient functions
func = @(x)(sum(100*(x(1:2:n).^2-x(2:2:n)).^2+(1-x(1:2:n)).^2));
grad = @(x)(rosen_ipopt_grad(x));

%
% Initial values
%
x           = zeros(n,1);
opts.print  = 1;
opts.maxIt  = 1000;
opts.tol    = 1e-3;
opts.whichL = 'wolfeG'; % wolfeG, wolfeB, mt 


% Call the solver
[x,fx,ex] = compLS(x,func,grad,opts);