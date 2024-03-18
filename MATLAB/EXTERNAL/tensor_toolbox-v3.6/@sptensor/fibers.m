function Xks = fibers(X,k,fsubs,Xfidx)
%FIBERS Extracts specified mode-k fibers and creates matrix.
%
%   M = FIBERS(X,K,FSUBS) extracts specified mode-K fibers from X and
%   assembles them into a matrix. The samples are specified by the tuples
%   in FSUBS. If X is a D-way tensor, then FSUBS is the list of
%   (D-1)-tuples specifying the fibers in mode K, arranged as an S x (D-1)
%   matrix. If mode-K of X is of size NK, then the result M is a matrix of
%   size NK x S. This is a submatrix of the full mode-K unfolding of X.
%
%   M = FIBERS(X,K,FSUBS,XFIDX) is the same as above except XFIDX is the
%   precomputed mode-K fiber indices of the nonzeros in X, i.e., 
%
%     XFIDX = FINDICES(X,K);
%
%   Precomputation is recommended for repeated called to FIBERS. 
%
%   See also SPTENSOR, CP_ARLS_LEV, SPTENMAT, FINDICES, TT_SUB2IND64.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

% Adapted from work by Casey Battaglino, circa 2016


% The orientation was selected because it is more efficient in memory than
% the traspose under the following assumptions...  
% * X is sparse,
% * NK >> S,
% * X is so sparse that many of the sampled fibers are all zeros


%%
d = ndims(X);
n = size(X); % Tensor dimensions
s = size(fsubs,1); % Number of samples

%% 

% Compute linear indices for the sampled fibers 
fidx = tt_sub2ind64(n([1:k-1,k+1:d]), fsubs);
[fidx_unique, ~, unique2repeat] = unique(fidx);

% Extract fiber indices if they weren't passed in
if nargin < 4    
    Xfidx = findices(X,k);
end

% Extract the row index and value for every nonzero whose mode-k fiber
% index that is in the sample set 
[tf,loc] = ismember(Xfidx, fidx_unique);

ii_unique = X.subs(tf, k); % Extract the mode-k index
jj_unique = loc(tf); % Extract the mode-k fiber index
vv_unique = X.vals(tf); % Extract the corresponding value
jj_unique = double(jj_unique);
Xsamp_unique = sparse(ii_unique, jj_unique, vv_unique, n(k), s);

% Convert back
Xks = Xsamp_unique(:,unique2repeat);


 
