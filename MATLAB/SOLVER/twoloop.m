function [p] = twoloop(gam,g,S,Y,idx)
%twoloop Computes the search direction using the L-BFGS two-loop recursion
%   [p] = twoloop(gam,g,S,Y,idx) uses the indices idx to access
%   the limited memory vectors s_{k-i} and y_{k-i}. 
%   This method computes 
%   
%       p = Hg
%   
%   where g is the negative gradient, H0 = gam.*I and S and Y hold
%   the limited-memory vectors
%
%--------------------------------------------------------------------------
% 02/21/24, J.B., initial implementation
%

ll      = length(idx);
alp     = zeros(ll,1);
rho     = zeros(ll,1);

% Forward loop
q = g;
for i=ll:-1:1
    r       = 1/(S(:,idx(i))'*Y(:,idx(i)));
    a       = r*(S(:,idx(i))'*q);
    q       = q - a.* Y(:,idx(i));
    alp(i)  = a;
    rho(i)  = r; 
end

% Initial matrix
p = gam.*q;

% Backward loop
for i=1:ll
    b       = rho(i)*(Y(:,idx(i))'*p);
    p       = p - (b-alp(i))*S(:,idx(i));
end

