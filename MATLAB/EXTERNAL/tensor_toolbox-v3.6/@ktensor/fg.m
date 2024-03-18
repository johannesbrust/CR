function [F,G] = fg(M,X,S,XnormSqr,vecG,onlyG,errCheck)
%FG CP decomposition function/gradient for ktensor with unit weights.
%
%   F = FG(M,X) calculates F(M) = ||M - X||^2 / ||X||^2 where M is a
%   ktensor (with unit weights) and X is a tensor of the same size.  
%
%   [F,G] = FG(M,X) also calculates the gradients. These are returned, by
%   default, as a cell array of matrices so that G{k} is the derivative
%   with respect to the k-th factor matrix, i.e., M.A{k}.
%
%   [F,G] = FG(M,X,S) calculates instead F(M) = ||M - X||^2 / S where S is
%   a scalar value.
%
%   [F,G] = FG(M,X,S,XNORMSQR) calculates instead 
%   F(M) = (XNORMSQR - 2*<X,M> + ||M||^2)/S, which is equal to ||M-X||^2/S
%   if XNORMSQR=||X||^2.
%
%   [F,G] = FG(M,X,S,XNORMSQR,VECG) returns G as a vector if VECG=true,
%   corresponding to the vector produced by tovec(M,false).
%
%   G = FG(M,X,S,XNORMSQR,VECG,ONLYG) returns only the gradient if ONLYG is
%   true.
%
%   See also cp_opt, ktensor/tovec.

%% Hidden option
%
%  [F,G] = fg(M,X,S,XNORMSQR,VECG,ONLYG,CHECKM) skips the unit weight check
%  on M if CHECKM=false.
%

%% Setup
if nargin < 7
    if ~exist('S','var')
        XnormSqr = norm(X).^2;
        S = XnormSqr;
    end
    if ~exist('XnormSqr','var')
        XnormSqr = norm(X).^2;
    end
    if ~exist('vecG','var')
        vecG = false;
    end
    if ~exist('onlyG','var')
        onlyG = false;
    end
    if ~exist('errCheck','var')
        errCheck = true;
    end
end

%% Check M has unit weights b/c only compute gradient wrt factor matrices.
if errCheck
    if ~all(M.lambda == 1)
        error('Ktensor M must have unit weights');
    end
    if ~isequal(size(X),size(M))
        error('Tensor X must has same size as Ktensor M');
    end
end

%% Extra various things
A = M.u;
d = ndims(M);
r = ncomponents(M);
computeG = (nargout > 1) || (onlyG);

%% Upsilon and Gamma
Upsilon = cell(d,1);
for k = 1:d
    Upsilon{k} = A{k}'*A{k};
end

if computeG
    Gamma = cell(d,1);
    for k = 1:d
        Gamma{k} = ones(r,r);
        for ell = [1:k-1,k+1:d]
            Gamma{k} = Gamma{k} .* Upsilon{ell};
        end
    end
    W = Gamma{1} .* Upsilon{1};
else
    W = ones(r,r);
    for k = 1:d
        W = W .* Upsilon{k};
    end
end


%% Calculate F2 = innerprod(M,X) in a way that sets up for G
U = mttkrp(X,A,1);
V = A{1} .* U;
F2 = sum(V(:));

%% Calculate G
if computeG
    G = cell(d,1);
    G{1} = -U + A{1}*Gamma{1};
    for k = 2:d
        U = mttkrp(X,A,k);
        G{k} = -U + A{k}*Gamma{k};
    end
    G = cellfun(@(x) x.*(2/S), G, 'UniformOutput', false);
    if vecG
        G = cell2mat(cellfun(@(x) x(:), G, 'UniformOutput', false));
    end
end

%% Calculate F
if onlyG
    F = G;
else
    % F1 = ||X||^2
    F1 = XnormSqr;   

    % F3 = ||M||^2
    F3 = sum(W(:));    

    % SUM
    F = (F1 - 2*F2 + F3)/S;
end

