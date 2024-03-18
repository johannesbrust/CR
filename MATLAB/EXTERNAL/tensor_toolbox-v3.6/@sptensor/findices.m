function fidx = findices(X,k)
%FINDICES Compute mode-k unfolding column index for every nonzero.
%
%   FIDX = FINDICES(X,K) computes the mode-k fiber indices (i.e., the
%   mode-k unfolding column indices) of every nonzero in X. 
%
%   See also SPTENSOR, SPTENMAT, FIBERS, TT_SUB2IND64.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

%%
d = ndims(X);
sz = size(X);
fidx = tt_sub2ind64(sz([1:k-1,k+1:d]), X.subs(:, [1:k-1,k+1:d]));
  