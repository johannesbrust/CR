function [fw,gw] = polyobji(w,x,y)
%POLYOBJi Computes the function and gradient for one data pair
%   
%   This implementation uses a large data matrix X
% 
%--------------------------------------------------------------------------
% 03/12/24, J.B., initial implementation

d = length(w);
N = length(x);
X = zeros(N,d);

% Polynomial data/ truncated Vandermonde
for i=d:-1:1
    X(:,i) = x.^(i-1);
end

% Residual
r = X*w-y;

% Loass and gradient
% fw = (r'*r)/N;
% gw = X'*((2/N)*r);

fw = (r'*r);
gw = X'*(2*r);
