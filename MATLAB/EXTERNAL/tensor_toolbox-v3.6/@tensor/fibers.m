function Xks = fibers(X,k,midx)
%FIBERS Extracts specified mode-k fibers and creates matrix.
%
%   M = FIBERS(X,K,FSUBS) extracts specified mode-K fibers from X and
%   assembles them into a matrix. The samples are specified by the tuples
%   in FSUBS. If X is a D-way tensor, then FSUBS is the list of
%   (D-1)-tuples specifying the fibers in mode K, arranged as an S x (D-1)
%   matrix. If mode-K of X is of size NK, then the result M is a matrix of
%   size NK x S. This is a submatrix of the full mode-K unfolding of X.
%
%   See also TENSOR, CP_ARLS_LEV, TT_SUB2IND64.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

% Adapted from work by Casey Battaglino, circa 2016

% The orientation was selected because it is more efficient in memory than
% the transpose under the following assumptions...  
% * X is sparse,
% * NK >> S,
% * X is so sparse that many of the sampled fibers are all zeros


%%
d = ndims(X);
n = size(X); % Tensor dimensions
s = size(midx,1); % Number of samples


% Dense Tensor

% Prepare to expand fiber multi-indices into tensor multi-indices
midx_samp = [midx(:, 1:k-1), zeros(s, 1), midx(:, k:d-1)];

% Create n(k) copies
midx_samp = kron(midx_samp, ones(n(k), 1)); % portable

% Insert indices for mode k
midx_samp(:,k) = repmat((1:n(k))', s, 1);

% Convert to linear indices
lidx_samp = tt_sub2ind64(n, midx_samp);

Xks = reshape(X.data(lidx_samp), n(k), []);

 
