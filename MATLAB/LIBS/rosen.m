function [f g] = rosen(x,varargin)
% Rosenbrock function 
[n,m] = size(x);
if nargin==1    % compute function value    
    
    if n<m
        x=x';
        n=m;
    end
    if mod(n,2)
        error('problem dimension is not an even number!');
    end
    f1 = x(2:2:n)-x(1:2:n-1).^2;
    f2 = 1-x(1:2:n-1);
    f = sum(f1.^2+f2.^2);
    if nargout>1 % compute gradient also
        g=zeros(n,1);
        g(2:2:n) = 2*f1;
        g(1:2:n-1) = -4*x(1:2:n-1).*f1 - 2*f2;
    end
    
else            % compute only gradient
    f1 = x(2:2:n)-x(1:2:n-1).^2;
    f2 = 1-x(1:2:n-1);
    f=zeros(n,1);
    f(2:2:n) = 2*f1;
    f(1:2:n-1) = -4*x(1:2:n-1).*f1 - 2*f2;
end