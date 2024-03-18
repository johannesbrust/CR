function [I,C,W,sz] = indices(varargin)
%INDICES Compute unique indices of a symmetric tensor.
%
%   [I,C,W,Q] = INDICES(A) returns all unique indices for a
%   symmetric tensor. Each row of I is an index listed in increasing order.
%   Each row of C is the corresponding monomial representation, and W 
%   is the count of how many times that index appears in the symmetric 
%   tensor. Q is the number of rows of I, the number of unique indices.
%
%   See also SYMTENSOR.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>
%
%   Updated on 07/2021
%      by Joao M. Pereira - https://github.com/joaompereira


if nargin == 0
    error('INDICES requires at least one input argument');
elseif nargin == 2 % Specify m and n      
    m = varargin{1};
    n = varargin{2};    
elseif nargin == 1
    A = varargin{1};
    if ~(isa(A,'symktensor') || isa(A,'tensor') || isa(A,'symtensor'))
        error('First argument must be a scalar or a tensor-like class');
    end
    m = ndims(A);
    n = size(A,1);
else
    error('Wrong number of input arguments');
end

%% Determine size
sz = nchoosek(m+n-1,m);

%% Compute I using MATLAB's nchoosek
I = nchoosek(1:n+m-1,m)-(0:m-1);

%% Compute C from I
if nargout>1
    C = double(I(:,1) == 1:n);
    for i=2:m
       C = C + (I(:,i) == 1:n);
    end
end

%% COMPUTE W (weights) from C
if nargout>2
    W = factorial(m)*ones(sz,1);

    for i=2:m
        W = W ./ sum(I(:,1:i)==I(:,i),2);
    end
end

end
