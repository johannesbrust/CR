function [g] = rosen_ipopt_grad(x)
% Rosenbrock function 
[n,m]       = size(x);

f1          = x(2:2:n)-x(1:2:n-1).^2;
f2          = 1-x(1:2:n-1);
g           = zeros(n,1);
g(2:2:n)    = 200*f1;
g(1:2:n-1)  = -400*x(1:2:n-1).*f1 - 2*f2;
    
end